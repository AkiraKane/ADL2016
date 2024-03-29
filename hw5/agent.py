import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import cv2

from base import BaseModel
from history import History
from ops import linear, conv2d
from replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl
from config import get_config

class Agent(BaseModel):
    def __init__(self, sess, min_action_set):
        config = get_config()
        super(Agent, self).__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'
        
        self.min_action_set = min_action_set
        self.action_size = self.min_action_set.shape[0]
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)
        # for playing
        self.play_history = History(self.config)
        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)
		
        self.windowname = 'RL'
        #cv2.startWindowThread()
	#cv2.namedWindow('RL')

        self.build_dqn()

    def build_dqn(self):
        """
        # TODO
            You need to build your DQN here.
            And load the pre-trained model named as './best_model.ckpt'.
            For example, 
                saver.restore(self.sess, './best_model.ckpt')
        """
        self.w = {}
        self.t_w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32', [None, self.history_length, self.screen_height, self.screen_width], name='s_t')

            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
                32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size, name='q')

            self.q_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in xrange(self.action_size):
                q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.merge_summary(q_summary, 'q_summary')

        # target network
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
                32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
            self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
                64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
                64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

            shape = self.target_l3.get_shape().as_list()
            self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                linear(self.target_l4, self.action_size, name='target_q')

            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted
            self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                tf.train.exponential_decay(
                    self.learning_rate,
                    self.learning_rate_step,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay,
                    staircase=True))
            self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

            self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

        tf.initialize_all_variables().run()

        self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

        self.load_model()
        self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w.keys():
          self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
   
   
    def train(self, env):
        start_step = self.step_op.eval()
        start_time = time.time()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []
        screen = env.new_game()
        #screen, reward, action, terminal = self.env.new_random_game()

        for _ in range(self.history_length):
            self.history.add(screen)
        
        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. predict
            action = self.predict(self.history.get())
            #print (action)
            # 2. act
            state = env.train_act(action)

            reward, screen, terminal = state[1]
            original_screen = state[0]
	    #cv2.imshow(self.windowname, original_screen)
            #cv2.waitKey(0)
            
            # 3. observe
            self.observe(screen, reward, action, terminal)
            
            if terminal:
                #screen, reward, action, terminal = env.new_random_game()
                screen = env.new_game()
                reward = action = 0
                _, _, terminal = env.state()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward
            
            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                        % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        self.save_model(self.step + 1)

                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)
              
                    if self.step > 180:
                        self.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of game': num_game,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
                        }, self.step)

                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []
  
    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
        self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step)
    
    def getSetting(self):
        """
        # TODO
            You can only modify these three parameters.
            Adding any other parameters are not allowed.
            1. action_repeat: number of time for repeating the same action 
            2. random_init_step: number of randomly initial steps
            3. screen_type: return 0 for RGB; return 1 for GrayScale
        """
        action_repeat = 4
        #random_init_step = 30
        screen_type = 0
        #return action_repeat, random_init_step, screen_type
        return action_repeat, screen_type
    
    def predict(self, s_t, test_ep = None):
        ep = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end)
                * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            action = random.randrange(self.action_size)

        else:
            action = self.q_action.eval({self.s_t: [s_t]})

        return action

    def observe(self, screen, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))

        self.history.add(screen)
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

        if self.step % self.target_q_update_step == self.target_q_update_step - 1:
            self.update_target_q_network()
    
    def q_learning_mini_batch(self):
        if self.memory.count < self.history_length:
            return
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        t = time.time()
        
        q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step,
        })

        self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def play(self, screen):
        '''
        for _ in range(self.history_length):
            test_history.add(screen)
        for t in tqdm(range(n_step), ncols=70):
            # 1. predict
            action = self.predict(test_history.get(), test_ep)
            # 2. act
            screen, reward, terminal = self.env.act(action, is_training=False)
            # 3. observe
            test_history.add(screen)
        '''
        """
        # TODO
            The "action" is your DQN argmax ouput.
            The "min_action_set" is used to transform DQN argmax ouput into real action number.
            For example,
                 DQN output = [0.1, 0.2, 0.1, 0.6]
                 argmax = 3
                 min_action_set = [0, 1, 3, 4]
                 real action number = 4
        """
        if not np.any(self.play_history.get()):
            for _ in range(self.history_length):
                self.play_history.add(screen)
        else:
            self.play_history.add(screen)

        action = self.predict(self.play_history.get(), test_ep=self.ep_end)
        
        #return self.min_action_set[action]
        return action
