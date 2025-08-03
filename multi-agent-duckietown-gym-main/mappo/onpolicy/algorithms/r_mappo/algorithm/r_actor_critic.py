import torch
import torch.nn as nn
from mappo.onpolicy.algorithms.utils.util import init, check
from mappo.onpolicy.algorithms.utils.cnn import CNNBase
from mappo.onpolicy.algorithms.utils.mlp import MLPBase
from mappo.onpolicy.algorithms.utils.rnn import RNNLayer
from mappo.onpolicy.algorithms.utils.act import ACTLayer
from mappo.onpolicy.algorithms.utils.popart import PopArt
from mappo.onpolicy.utils.util import get_shape_from_obs_space
import mappo.onpolicy.utils.graphUtils.graphML as gml


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space,  device=torch.device("cpu"), graph_ml=False, attention =False):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self.device = device
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.graph_ml = graph_ml
        self.num_agents = args.num_agents
        self.episode_length = args.episode_length
        self.attention = attention

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.discrete = args.discrete

        if self.graph_ml:
            self.S = None
            nGraphFilterTaps = [args.gnn_filter_taps]
            nAttentionHeads = [args.n_attention_heads]
            dimNodeSignals = [2 ** 7]
            # dimNodeSignals = [64, 2 ** 7]
            # dimNodeSignals = [256, 2 ** 7]
            numCompressFeatures = [128, 128]

            self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
            self.F = [numCompressFeatures[-1]] + dimNodeSignals  # Features
            self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
            self.P = nAttentionHeads
            self.E = 1  # Number of edge features
            self.bias = True

            gfl = []  # Graph Filtering Layers
            for l in range(self.L):
                # \\ Graph filtering stage:
                if self.attention:
                    gfl.append(gml.GraphFilterBatchAttentional(self.F[l], self.F[l + 1], self.K[l],
                                                               self.P[l], self.E, self.bias,
                                                               concatenate=False,attentionMode='KeyQuery'))
                else:
                    gfl.append(gml.GraphFilterBatchGSO(self.F[l], self.F[l + 1], self.K[l],
                                                               self.E, self.bias))
                    # \\ Nonlinearity
                    gfl.append(nn.ReLU(inplace=True))
                # There is a 2*l below here, because we have three elements per
                # layer: graph filter, nonlinearity and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.

            # And now feed them into the sequential
            self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers


        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, self.discrete)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.graph_ml:
            actor_features = actor_features.permute([1, 0]).unsqueeze(0)
            for l in range(self.L):
                # \\ Graph filtering stage:
                # There is a 3*l below here, because we have three elements per
                # layer: graph filter, nonlinearity and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter
            actor_features = self.GFL(actor_features)
            actor_features = actor_features[0].permute(1, 0)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, gso=None, agent_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)


        if self.graph_ml:
            extractFeatureMap = torch.zeros(self.episode_length, actor_features.shape[1], self.num_agents).to(**self.tpdv)
            for agent_i in range(self.num_agents):
                extractFeatureMap[:, :, agent_i] = actor_features[agent_masks==agent_i]
            gso = gso[0:self.episode_length]
            gso = gso.unsqueeze(1)
            # actor_features = actor_features.permute([1, 0]).unsqueeze(0)
            # actor_features = actor_features.unsqueeze(2)
            for l in range(self.L):
                # \\ Graph filtering stage:
                # There is a 3*l below here, because we have three elements per
                # layer: graph filter, nonlinearity and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                self.GFL[2 * l].addGSO(gso)  # add GSO for GraphFilter
            extractFeatureMap = self.GFL(extractFeatureMap)

            for agent_i in range(self.num_agents):
                actor_features[agent_masks==agent_i] = extractFeatureMap[:, :, agent_i]
            # actor_features = actor_features[0].permute(1, 0)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

    def addGSO(self, S):
        S = torch.tensor(S).to(self.device)
        S = S.unsqueeze(0)
        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            # assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

        # if self.config.GSO_mode == 'dist_GSO_one':
        #     self.S[self.S > 0] = 1
        # elif self.config.GSO_mode == 'full_GSO':
        #     self.S = torch.ones_like(self.S).to(self.config.device)
        # self.S[self.S > 0] = 1


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states



