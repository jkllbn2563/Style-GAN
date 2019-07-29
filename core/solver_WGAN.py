import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import sys

from utils import *
sys.path.append('./coco-caption')
from bleu import evaluate,evaluate_captions,evaluate_for_particular_captions
from discriminator_WGAN import Discriminator
from rollout import ROLLOUT
from dataloader import Dis_dataloader


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17) 
                - image_idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.dis_batch_size = kwargs.pop('dis_batch_size', 30)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.rollout_num = kwargs.pop('rollout_num', 10)
        self.dis_dropout_keep_prob = kwargs.pop('dis_dropout_keep_prob', 1.0)
        self.data_path = kwargs.pop('data_path', './data/')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self):
        data_save_path = self.data_path

        #####using just sentiment captions
        sentiment_i = np.where( self.data['captions'][:, 3] != 0 )[0]
        captions = self.data['captions'][sentiment_i,:21]
        n_examples = captions.shape[0]
        n_iters_per_epoch = int(np.floor( float(n_examples) / self.batch_size ) ) 
        image_idxs = self.data['image_idxs'][sentiment_i]

        ###### using all captions
        #captions = self.data['captions'][:,:21]
        #n_examples = captions.shape[0]
        #n_iters_per_epoch = int(np.floor(float(n_examples) / self.batch_size))
        #image_idxs = self.data['image_idxs']

        features_temp_1 = np.zeros([self.data['features'].shape[0], 49, 2052])
        features_temp_2 = self.data['features'].reshape(-1,49,2064)
        features_temp_1[:, :, 0:512] =  features_temp_2[:,:,0:512]
        features_temp_1[:, :, 512:1024] = features_temp_2[:, :, 516:1028]
        features_temp_1[:, :, 1024:1536] = features_temp_2[:, :, 1028:1540]
        features_temp_1[:, :, 1536:2048] = features_temp_2[:, :, 1540:2052]
        features = features_temp_1

	features_temp_1 = np.zeros([self.val_data['features'].shape[0], 49, 2052])
        features_temp_2 = self.val_data['features'].reshape(-1, 49, 2064)
        features_temp_1[:, :, 0:512] = features_temp_2[:, :, 0:512]
        features_temp_1[:, :, 512:1024] = features_temp_2[:, :, 516:1028]
        features_temp_1[:, :, 1024:1536] = features_temp_2[:, :, 1028:1540]
        features_temp_1[:, :, 1536:2048] = features_temp_2[:, :, 1540:2052]
        val_features = features_temp_1

        n_iters_val = int(np.ceil(float(val_features.shape[0]) / self.batch_size))

        # build graphs for training model and sampling captions
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=self.model.T-4)

        # train op
        with tf.variable_scope(tf.get_variable_scope()):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            params = [param for param in tf.trainable_variables() if not('discriminator' in param.name)]
            grads = tf.gradients(loss, params)
            grads_and_vars = list(zip(grads, params)) #tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            print var.op.name,'ooooo'
            tf.summary.histogram(var.op.name + '/gradient', grad)

        summary_op = tf.summary.merge_all()

        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True

        dis_embedding_dim = 256
        dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, self.model.T-4]
        dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        dis_l2_reg_lambda = 0.2

        discriminator = Discriminator(sequence_length=self.model.T-4, num_classes=2, vocab_size=self.model.V,
                                      embedding_size=dis_embedding_dim,
                                      filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                      l2_reg_lambda=dis_l2_reg_lambda)

        rollout = ROLLOUT(self.model, 0.8)

        dis_data_loader = Dis_dataloader(self.dis_batch_size)

        rewards = np.zeros((self.batch_size, self.model.T-4), dtype=np.float32)

	dis_results_file = open( os.path.join( self.model_path, 'dis_results_file_4.txt' ), 'w' )

        with tf.Session(config=config) as sess:

            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter( self.log_path, graph=tf.get_default_graph() )
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            print 'Start pre-training...'

            for e in range(0):#self.n_epochs):
            
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]
           
                for i in range(n_iters_per_epoch):

                    captions_batch = captions[i * self.batch_size:(i + 1) * self.batch_size]
                    image_idxs_batch = image_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                    features_batch = features[image_idxs_batch]

                    feed_dict = { self.model.whole_samples: captions_batch[:,4:self.model.T], self.model.rewards: rewards,
                                  self.model.features: features_batch,
                                  self.model.captions: captions_batch, self.model.mode_learning: 1 }
            
                    _, l = sess.run( [train_op, loss], feed_dict )

                    curr_loss += l

                    if (i + 1) % self.print_every == 0:
                        # print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, l)
            
                        ground_truths = captions[image_idxs == image_idxs_batch[0], 4:]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j + 1, gt)
                        feed_dict = { self.model.features: features_batch,
                                      self.model.whole_samples: captions_batch[:,4:self.model.T],
                                      self.model.nsample: 0, self.model.mode_sampling: 1,
                                      self.model.captions: captions_batch }
            
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" % decoded[0]

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                captions_batch = captions[0 * self.batch_size:(0 + 1) * self.batch_size]
                if self.print_bleu:

                    all_gen_cap = np.ndarray( (val_features.shape[0], self.model.T-4) )
                    pos = [1]
                    neg = [-1]

                    val_features[:, :, 2048:2052] = [0, 1, 0, 1]

                    for i in range(n_iters_val):
                        features_batch = val_features[i * self.batch_size:(i + 1) * self.batch_size]
                        feed_dict = {self.model.features: features_batch,
                                     self.model.whole_samples: captions_batch[:,4:self.model.T],
                                     self.model.nsample: 0, self.model.mode_sampling: 1,
                                     self.model.captions: captions_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, os.path.join(data_save_path, "val/val.candidate.captions.pkl"))
                    scores = evaluate(data_path=data_save_path, split='val', get_scores=True)

                    print "scores_pos==================", scores
                    write_bleu(scores=scores, path=self.model_path, epoch=e, senti=pos)

                    val_features[:, :, 2048:2052] = [0, 0, 1, 2]

                    for i in range(n_iters_val):
                        features_batch = val_features[i * self.batch_size:(i + 1) * self.batch_size]
                        feed_dict = {self.model.features: features_batch,
                                     self.model.whole_samples: captions_batch[:,4:self.model.T],
                                     self.model.nsample: 0, self.model.mode_sampling: 1,
                                     self.model.captions: captions_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, os.path.join(data_save_path, "val/val.candidate.captions.pkl"))
                    scores = evaluate(data_path=data_save_path, split='val', get_scores=True)
                    print "scores_neg==================", scores
                    write_bleu(scores=scores, path=self.model_path, epoch=e, senti=neg)

                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step = e+1)
                    print "model-%s saved." % (e + 1)

            print 'Start pre-training discriminator...'
            for e in range(0):#self.n_epochs):

                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]
                dis_loss=0
                for i in range(n_iters_per_epoch):

                    captions_batch = captions[i * self.batch_size:(i + 1) * self.batch_size]
                    image_idxs_batch = image_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                    features_batch = features[image_idxs_batch]

                    feed_dict = { self.model.features: features_batch,
                                  self.model.whole_samples: captions_batch[:, 4:self.model.T], self.model.nsample: 0,
                                  self.model.mode_sampling: 1,
                                  self.model.captions: captions_batch }

                    for d_step in range(3):
                    	negative_file = sess.run(generated_captions, feed_dict=feed_dict)
                    	positive_file = captions_batch[:, 4:self.model.T]
                    	dis_data_loader.load_train_data(positive_file, negative_file)
                    	for it in xrange(dis_data_loader.num_batch):
                        	x_batch, y_batch = dis_data_loader.next_batch()
                        	feed = {
                            	discriminator.input_x: x_batch,
                            	discriminator.input_y: y_batch,
                            	discriminator.dropout_keep_prob: self.dis_dropout_keep_prob
                        	}
				dis_l = sess.run(discriminator.loss, feed)
                        	dis_loss = dis_loss + dis_l
                        	_ = sess.run(discriminator.train_op, feed)
                        	_ = sess.run(discriminator.params_clip, feed)
		   
                dis_results_file.write('The loss in epoch %i is %f \n' %(e+1, dis_loss) )
                dis_results_file.flush()

	    	saver.save(sess, os.path.join(self.model_path, 'model_and_dis'), global_step = e+1)

            print '#########################################################################'
            print 'Start Adversarial Training...'
            for e in range(10):

                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):

                    captions_batch = captions[i * self.batch_size:(i + 1) * self.batch_size]
                    image_idxs_batch = image_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                    features_batch = features[image_idxs_batch]

                    feed_dict = {self.model.features: features_batch, self.model.whole_samples: captions_batch[:,4:self.model.T],
                                 self.model.nsample: 0, self.model.mode_sampling: 1, self.model.captions: captions_batch}
                    samples_whole = sess.run(generated_captions, feed_dict=feed_dict)

                    rewards = rollout.get_reward(sess, samples_whole, generated_captions, self.rollout_num, discriminator, features_batch, captions_batch)

                    feed_dict = {self.model.whole_samples: samples_whole, self.model.rewards: rewards,
                                 self.model.features: features_batch,
                                 self.model.captions: captions_batch, self.model.mode_learning: 2}
                    _, l_reward = sess.run([train_op, loss], feed_dict=feed_dict)
                    curr_loss += l_reward

                    feed_dict = {self.model.features: features_batch,
                                 self.model.whole_samples: captions_batch[:, 4:self.model.T], self.model.nsample: 0,
                                 self.model.mode_sampling: 1,
                                 self.model.captions: captions_batch}
                    
                    for d_step in range(3):
                    	negative_file = sess.run(generated_captions, feed_dict=feed_dict)
                    	positive_file = captions_batch[:, 4:self.model.T]
                    	dis_data_loader.load_train_data(positive_file, negative_file)
                    	for it in xrange(dis_data_loader.num_batch):
                        	x_batch, y_batch = dis_data_loader.next_batch()
                        	feed = {
                                    	discriminator.input_x: x_batch,
                                    	discriminator.input_y: y_batch,
                                    	discriminator.dropout_keep_prob: self.dis_dropout_keep_prob
                                	}
                        	_ = sess.run(discriminator.train_op, feed)
                        	_ = sess.run(discriminator.params_clip, feed)


                    if (i + 1) % self.print_every == 0:

                        ground_truths = captions[image_idxs == image_idxs_batch[0], 4:]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j + 1, gt)
                        feed_dict = {self.model.features: features_batch,
                                     self.model.whole_samples: captions_batch[:,4:self.model.T],
                                     self.model.nsample: 0, self.model.mode_sampling: 1,
                                     self.model.captions: captions_batch}
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" % decoded[0]

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                captions_batch = captions[0 * self.batch_size:(0 + 1) * self.batch_size]
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], self.model.T-4))
                    #######################################\sentiCap

                    pos = [1]
                    neg = [-1]

                    val_features[:, :, 2048:2052] = [0, 1, 0, 1]

                    for i in range(n_iters_val):
                        features_batch = val_features[i * self.batch_size:(i + 1) * self.batch_size]
                        feed_dict = {self.model.features: features_batch,
                                     self.model.whole_samples: captions_batch[:,4:self.model.T],
                                     self.model.nsample: 0, self.model.mode_sampling: 1, self.model.captions: captions_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, os.path.join(data_save_path, "val/val.candidate.captions.pkl"))
                    scores = evaluate(data_path=data_save_path, split='val', get_scores=True)

                    print "scores_pos==================", scores

                    write_bleu(scores=scores, path=self.model_path, epoch=e, senti=pos)

                    val_features[:, :, 2048:2052] = [0, 0, 1, 2]

                    for i in range(n_iters_val):

                        features_batch = val_features[i * self.batch_size:(i + 1) * self.batch_size]
                        feed_dict = {self.model.features: features_batch, self.model.whole_samples: captions_batch[:,4:self.model.T],
                                     self.model.nsample: 0, self.model.mode_sampling: 1, self.model.captions: captions_batch}

                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, os.path.join(data_save_path, "val/val.candidate.captions.pkl"))
                    scores = evaluate(data_path=data_save_path, split='val', get_scores=True)
                    print "scores_neg==================", scores
                    write_bleu(scores=scores, path=self.model_path, epoch=e, senti=neg)

                if (e + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model_adv'), global_step = e+1)
                    print "model-%s saved." % (e + 1)


    def test(self, data, split='train', attention_visualization=False, save_sampled_captions=False, senti=[0]):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        max_len_captions = 20

        features_temp_1 = np.zeros([data['features'].shape[0], 49, 2052])
        features_temp_2 = data['features'].reshape(-1, 49, 2064)

        features_temp_1[:, :, 0:512] = features_temp_2[:, :, 0:512]
        features_temp_1[:, :, 512:1024] = features_temp_2[:, :, 516:1028]
        features_temp_1[:, :, 1024:1536] = features_temp_2[:, :, 1028:1540]
        features_temp_1[:, :, 1536:2048] = features_temp_2[:, :, 1540:2052]

        features = features_temp_1
        captions = data['captions'][:, :21]

        if senti == [1]:
            data_save_path = '/linux_data_drive/sentiment_dataset/sent_project/data_generation/data_7_pos_resNet-152/'
        else:
            data_save_path = '/linux_data_drive/sentiment_dataset/sent_project/data_generation/data_7_neg_resNet-152/'

        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.floor(float(n_examples) / self.batch_size))

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(
            max_len=max_len_captions)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            if self.print_bleu:
                all_gen_cap = np.ndarray((features.shape[0], max_len_captions))

                for i in range(n_iters_per_epoch):
                    features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                    captions_batch = captions[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch,
                                 self.model.whole_samples: captions_batch[:, 4:self.model.T],
                                 self.model.nsample: 0, self.model.mode_sampling: 1,
                                 self.model.captions: captions_batch}

                    gen_cap = sess.run(sampled_captions, feed_dict=feed_dict)
                    all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)

                save_pickle(all_decoded, os.path.join(data_save_path + 'test/test.candidate.captions.pkl'))
                scores = evaluate(data_path=data_save_path, split=split, get_scores=True)

            # if attention_visualization:
            #     for n in range(10):
            #         print "Sampled Caption: %s" % decoded[n]

                    # # Plot original image
                    # img = ndimage.imread(image_files[n])
                    # plt.clf()
                    # plt.subplot(4, 5, 1)
                    # plt.imshow(img)
                    # plt.axis('off')
                    #
                    # # Plot images with attention weights
                    # words = decoded[n].split(" ")
                    # for t in range(len(words)):
                    #     if t > 18:
                    #         break
                    #     plt.subplot(4, 5, t + 2)
                    #     plt.text(0, 1, '%s(%.2f)' % (words[t], bts[n, t]), color='black', backgroundcolor='white',
                    #              fontsize=8)
                    #     plt.imshow(img)
                    #     alp_curr = alps[n, t, :].reshape(14, 14)
                    #     alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                    #     plt.imshow(alp_img, alpha=0.85)
                    #     plt.axis('off')
                    # plt.savefig(str(n) + 'test.pdf')

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], max_len_captions))
                num_iter = int(np.floor(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch}
                    all_sam_cap[i * self.batch_size:(i + 1) * self.batch_size] = sess.run(sampled_captions, feed_dict)


                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))

    def inference(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './model/lstm/model-20')
            features_batch, image_files = sample_coco_minibatch_inference(data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)
            print "end"
            print decoded

