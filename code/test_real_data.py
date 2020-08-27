import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description="Pytorch eSL-Net")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", type=str, default="pre_training/model_epoch_57.pth", help="model path")

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
	raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"].module.cuda()


txt_path="../data_example/test.txt"

fh = open(txt_path, 'r')


avg_elapsed_time = 0.0
count = 0.0

for line in fh:
        count += 1
        print('count=',count)
        line = line.rstrip()
        words = line.split()
        img = Image.open(words[0])
        im_b_y = np.array(img)
		
        for f in range(0,20):
                im_input=im_b_y[np.newaxis,:,:]
                event_sequence = open(words[1], 'r')
                event_frame=np.zeros([40,im_b_y.shape[0],im_b_y.shape[1]],float)
                event_flag=np.zeros([im_b_y.shape[0],im_b_y.shape[1]],int)
                for e in event_sequence:
                        e = e.rstrip()
                        event = e.split()
                        if int(event[3])>0:
                                if event_flag[int(event[1])-1,int(event[2])-1]<40:
                                        if float(event[0])<f*0.05:
                                                event_frame[event_flag[int(event[1])-1,int(event[2])-1],int(event[1])-1,int(event[2])-1]=(-1-float(event[0])+f*0.05)
                                        else:
                                                event_frame[event_flag[int(event[1])-1,int(event[2])-1],int(event[1])-1,int(event[2])-1]=1-float(event[0])+f*0.05
                        else:
                                if event_flag[int(event[1])-1,int(event[2])-1]<40:
                                        if float(event[0])<f*0.05:
                                                event_frame[event_flag[int(event[1])-1,int(event[2])-1],int(event[1])-1,int(event[2])-1]=(1-f*0.05+float(event[0]))
                                        else:
                                                event_frame[event_flag[int(event[1])-1,int(event[2])-1],int(event[1])-1,int(event[2])-1]=-1+float(event[0])-f*0.05
                        event_flag[int(event[1])-1,int(event[2])-1]=event_flag[int(event[1])-1,int(event[2])-1]+1

                event_array=np.array(event_frame).astype(np.float32)
                im_input= np.append(im_input, event_array, axis = 0)

                im_input = Variable(torch.from_numpy(im_input).float(), volatile=True).view(1, im_input.shape[0], im_input.shape[1], im_input.shape[2]).cuda()
        
                start_time = time.time()
                HR = model(im_input)
                elapsed_time = time.time() - start_time
                avg_elapsed_time += elapsed_time

                HR = HR.cpu()
                im_h_y=np.squeeze(HR.data.numpy().astype(np.float32))
                im_h_y[im_h_y<0] = 0
                im_h_y[im_h_y>255.] = 255.
                image_hr = Image.fromarray(im_h_y.astype(np.uint8))
                image_hr_name=words[0]
                image_hr.save('../data_example/camerashake_results/'+ image_hr_name[-8:-4]+ '_'+ str(int(f))+'.png','PNG')


print("Model=", opt.model)
print("It takes average {}s for processing".format(avg_elapsed_time/(count*20)))

