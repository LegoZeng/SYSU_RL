import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import model


BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


class Trainer:

    def __init__(self, state_dim, action_dim, action_lim, replay_buffer):

        self.device = torch.device("cuda")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.replay_buffer = replay_buffer
        self.iter = 0
        self.noise = utils.Noise(self.action_dim)

        self.actor = model.Actor(
            self.state_dim, self.action_dim, self.action_lim).to(self.device)
        self.target_actor = model.Actor(
            self.state_dim, self.action_dim, self.action_lim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), LEARNING_RATE)

        self.critic = model.Critic(
            self.state_dim, self.action_dim).to(self.device)
        self.target_critic = model.Critic(
            self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), LEARNING_RATE)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        """
        gets the action without noise
        """
        state = torch.from_numpy(state).to(self.device)
        action = self.target_actor.forward(state).detach()
        return action.numpy()

    def get_exploration_action(self, state):
        """
        gets the action with noise
        """
        state = torch.from_numpy(state).to(self.device)
        action = self.actor.forward(state).detach()
        new_action = (action.cpu().numpy() +
                      self.noise.sample()) * self.action_lim
        return new_action

    def train(self, timestamp):
        s1, a1, r1, s2 = self.replay_buffer.sample(BATCH_SIZE)

        s1 = torch.from_numpy(s1).to(self.device)
        a1 = torch.from_numpy(a1).to(self.device)
        r1 = torch.from_numpy(r1).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)

        # optimize critic
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(
            self.target_critic.forward(s2, a2).detach())
        y_expected = r1 + GAMMA*next_val
        y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        loss_critic = F.mse_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # optimize actor
        pred_a1 = self.actor.forward(s1)
        loss_actor = -self.critic.forward(s1, pred_a1).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # avoid bootstrapping
        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)

        if self.iter % 50 == 0:
            with open(timestamp+'loss.txt', 'a') as f:
                f.write(
                    f'{self.iter}\t{loss_actor.data.cpu().numpy()}\t{loss_critic.data.cpu().numpy()}\n')
        self.iter += 1

    def save(self, episode_count, timestamp):
        """
        save the target actor and critic models
        """
        fpath = './'+timestamp+'Models/'
        torch.save(self.target_actor.state_dict(), fpath +
                   str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), fpath +
                   str(episode_count) + '_critic.pt')
        print('Models saved successfully')

    def load(self, episode, timestamp):
        """
        load and copy the actor and critic models
        """
        fpath = './'+timestamp+'Models'
        self.actor.load_state_dict(torch.load(
            fpath+'/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(
            fpath+'/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')
