from core.solver_WGAN import CaptioningSolver
from core.model_WGAN import CaptionGenerator
from core.utils import load_coco_data
import os

def main():

    switch = [1]

    if switch == [1]:
       data_save_path = './data_positive/'
    else:
       data_save_path = './data_negative/'

    image_save_path = './images/'
    save_path = './model_test/'
    log_save_path = './log/'
    save_path_pretrain = './model/'


    data = load_coco_data(data_path=data_save_path, split='train')
    word_to_idx = data['word_to_idx']
    val_data = load_coco_data( data_path=data_save_path, split='val' )
    test_data = load_coco_data( data_path=data_save_path, split='test' )

    model = CaptionGenerator( word_to_idx, dim_feature=[49, 2052], dim_embed=300,
                                       dim_hidden=512, dim_senti=512, n_time_step=20, beam_index=5, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True )

    solver = CaptioningSolver( model, data, val_data, n_epochs=0, batch_size=64, dis_batch_size=80, update_rule='adam',
                                          learning_rate=0.001, print_every=1, save_every=1, image_path=image_save_path,
                                    pretrained_model=None, model_path=save_path, test_model=os.path.join(save_path_pretrain+'model_15'),
                                     print_bleu=True, log_path=log_save_path, rollout_num=5, dis_dropout_keep_prob=1.0,
                               data_path=data_save_path )

    solver.test( test_data, 'test', senti=switch )

if __name__ == "__main__":
    main()
