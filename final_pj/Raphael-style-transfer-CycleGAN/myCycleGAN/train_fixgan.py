import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import torch.nn.functional as F 
import os 

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    RESOLUTIONS = [128,256,512]

    #如果每次上采样率接近4/3的话，从128开始
    #默认RESOLUTIONS = [128,256,512]
    #[128,172,232,312,416,512]

    MODELS = []

    #定义金字塔型层级模型
    for _i in range(len(RESOLUTIONS)):
        model = create_model(opt)      
        model.setup(opt)               
        MODELS.append(model)

    for model_idx,model in enumerate(MODELS):

        #定义模型分辨率用于保存等
        resolution = RESOLUTIONS[model_idx]

        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            model.update_learning_rate() 

            epoch_start_time = time.time() 
            iter_data_time = time.time()   
            epoch_iter = 0                  
            visualizer.reset()           

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration

                #使用上一个网络的输出作为输入
                for _i in range(model_idx-1):
                    pre_model.set_input(data)
                    size_of_img = (RESOLUTIONS[_i],RESOLUTIONS[_i])

                    if _i == 0:
                        data['B'] = F.interpolate(data['B'],size=size_of_img,mode='bilinear',align_corners=True)
                        data['B'] = data['B'].mean(axis=1)
                        data['B'] = torch.stack((data['B'],data['B'],data['B']),axis=1)
                        
                    #更改dataB并放入模型,获取最终的dataB                    
                    MODELS[_i].set_input(data)
                    data['B'] = MODELS[_i].fake_A.detach().clone()

                model.set_input(data)
                model.optimize_parameters()
                
                #计算迭代时间
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                #间隔一段时间进行可视化
                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()

                    print('try save img' ,resolution)
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result,resolution)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                
                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'

                    #添加不同resolution
                    model.save_networks(save_suffix + '_' + str(resolution))

                iter_data_time = time.time()

                if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                    
                    model.save_networks('latest' + '_' + str(resolution))
                    model.save_networks(str(epoch) + '_' + str(resolution))

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
