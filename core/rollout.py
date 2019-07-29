import numpy as np


class ROLLOUT(object):
    def __init__(self, model, update_rate):
        self.model = model
        self.update_rate = update_rate

    def get_reward(self, sess, input_x, generated_captions, rollout_num, discriminator, features_batch, GT_samples):
        rewards = []
        for i in range(rollout_num):


            for given_num in range(1, self.model.T-4):
                # print given_num, "----------------"

                feed_dict = {self.model.features: features_batch,
                             self.model.whole_samples: input_x, self.model.nsample: given_num,
                             self.model.mode_sampling: 2, self.model.captions: GT_samples}

                samples = sess.run(generated_captions, feed_dict=feed_dict)

                # print samples
                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])

                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                # completed sentence reward
                rewards[self.model.T-4 - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards

