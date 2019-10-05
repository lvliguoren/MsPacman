import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, fully_connected
from ReplayMemory import ReplayMemory
from PIL import Image, ImageDraw
import matplotlib.animation as animation
import matplotlib.pyplot as plt


env = gym.make("MsPacman-v0")
obs = env.reset()
mspacman_color = np.array([210, 164, 74]).mean()


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.mean(axis=2)
    img[img == mspacman_color] = 0
    img = (img - 128)/128 - 1
    return img.reshape(88, 80, 1)


input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"]*3
conv_activation = [tf.nn.relu]*3
n_hidden_in = 64*11*10
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n
initializer = tf.contrib.layers.variance_scaling_initializer()


def q_network(X_state, scope):
    prev_layer = X_state
    conv_layers = []
    with tf.variable_scope(scope) as scope:
        for n_maps, kernel_size, stride, padding, activation in zip(
            conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation
        ):
            prev_layer = convolution2d(prev_layer, num_outputs=n_maps, kernel_size=kernel_size, stride=stride,
                                       padding=padding, activation_fn=activation, weights_initializer=initializer)
            conv_layers.append(prev_layer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = fully_connected(last_conv_layer_flat, n_hidden, activation_fn=hidden_activation, weights_initializer=initializer)
        outputs = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
        return outputs, trainable_vars_by_name


X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
actor_q_values, actor_vars = q_network(X_state, "q_networks/actor")
critic_q_values, critic_vars = q_network(X_state, "q_networks/critic")

copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    # critic_q_values输出是该State下9种Action的Q值，q_value表示该State该Action下的Q值
    q_value = tf.reduce_sum(critic_q_values*tf.one_hot(X_action, n_outputs), axis=1, keep_dims=True)
    # 目标Q值
    y = tf.placeholder(tf.float32, shape=[None, 1])
    cost = tf.reduce_mean(tf.square(y-q_value))
    global_step = tf.Variable(0, trainable=False, name="global_step")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(cost, global_step=global_step)


replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)


def sample_memories(batch_size):
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000


def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)  # random action
    else:
        return np.argmax(q_values)  # optimal action


n_steps = 100000
training_start = 1000
training_interval = 3
save_steps = 50
copy_steps = 25
discount_rate = 0.95
skip_start = 90
batch_size = 50
iteration = 0
checkpoint_path = "./my_dqn.ckpt"
done = True
saver = tf.train.Saver()


with tf.Session() as sess:
    if os.path.isfile(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        if done:
            obs = env.reset()
            for skip in range(skip_start):
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        q_values = actor_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        if iteration < training_start or iteration % training_interval !=0:
            continue

        X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
        next_q_values = actor_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards+continues*discount_rate*max_next_q_values
        training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

        if step % copy_steps == 0:
            copy_critic_to_actor.run()

        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)


# frames = []
# n_max_steps = 10000
#
#
# def update_scene(num, frames, patch):
#     patch.set_data(frames[num])
#     return patch,
#
#
# def plot_animation(frames, repeat=False, interval=40):
#     plt.close()  # or else nbagg sometimes plots in the previous cell
#     fig = plt.figure()
#     patch = plt.imshow(frames[0])
#     plt.axis('off')
#     return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)
#
#
# with tf.Session() as sess:
#     saver.restore(sess, checkpoint_path)
#
#     obs = env.reset()
#     for step in range(n_max_steps):
#         state = preprocess_observation(obs)
#
#         # Online DQN evaluates what to do
#         q_values = actor_q_values.eval(feed_dict={X_state: [state]})
#         action = np.argmax(q_values)
#
#         # Online DQN plays
#         obs, reward, done, info = env.step(action)
#
#         img = env.render(mode="rgb_array")
#         frames.append(img)
#
#         if done:
#             break
#
#
# video = plot_animation(frames)
# plt.show()
# env.close()