import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description="Pytorch eSL-Net Eval")
parser.add_argument("--sr", default="1", type=int, help="with sr is 1, without sr is 0")
parser.add_argument("--model", type=str, default="pre_trained/model_withsr_pretrained.pt", help="model path")
parser.add_argument("--num_frame", default="3", type=int, help="number of output frames for one blur images")
parser.add_argument("--output_path", default="realdata_sr/", type=str, help="output path")

def PSNR(pred, gt, shave_border=0):
	height, width = pred.shape[:2]
	pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
	gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
	imdff = pred -gt
	rmse = math.sqrt(np.mean(imdff ** 2))
	if rmse == 0:
		return 100
	return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
num_frame=opt.num_frame

if opt.sr==1:
    from model_sr import Net
else:
    from model_withoutsr import Net

model = Net().cuda()
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.model).items()},strict=False)


txt_path="../data_example/realdata_list.txt"

fh = open(txt_path, 'r')

avg_elapsed_time = 0.0
count = 0.0

for line in fh:
        print('count=',count/num_frame)
        line = line.rstrip()
        words = line.split()
        img = Image.open(words[0])
        im_b_y = np.array(img)
        event_sequence=sio.loadmat(words[1])
        start_time=event_sequence['start_timestamp']
        end_time=event_sequence['end_timestamp']
        event_time=event_sequence['section_event_timestamp']
        event_polar=event_sequence['section_event_polarity']
        event_y=event_sequence['section_event_y']
        event_x=event_sequence['section_event_x']

        image_interval=(end_time-start_time)/(num_frame-1);
        frame_interval=(end_time-start_time)/20;

        image_hr_name=words[0]
        isexists = os.path.exists(opt.output_path + image_hr_name[16:-17])
        if not isexists:
            os.makedirs(opt.output_path + image_hr_name[16:-17])
		
        for f in range(0,num_frame):
                count += 1
                im_input=im_b_y[np.newaxis,:,:]
                event_frame=np.zeros([40,im_b_y.shape[0],im_b_y.shape[1]],int)
                for event_i in range(0,event_time.shape[1]):
                    if event_time[0,event_i]<image_interval*f+start_time:
                            cha = image_interval*f+start_time-event_time[0,event_i]
                            frame_index=cha//frame_interval
                            if frame_index==20:
                                frame_index=19
                            if event_polar[0,event_i]>0:
                                event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1]+1
                            else:
                                event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1]+1
                    else:
                            cha = event_time[0,event_i]-image_interval*f-start_time
                            frame_index=cha//frame_interval
                            if frame_index==20:
                                frame_index=19
                            if event_polar[0,event_i]>0:
                                event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2),event_y[0,event_i]-1,event_x[0,event_i]-1]+1
                            else:
                                event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(frame_index*2)+1,event_y[0,event_i]-1,event_x[0,event_i]-1]+1

                event_array=np.array(event_frame).astype(np.float32)
                im_input= np.append(im_input, event_array, axis = 0)

                with torch.no_grad():
                    im_input = Variable(torch.from_numpy(im_input).float(), volatile=True).view(1, im_input.shape[0], im_input.shape[1], im_input.shape[2]).cuda()
        
                    start = time.time()
                    HR = model(im_input)
                    elapsed_time = time.time() - start
                    avg_elapsed_time += elapsed_time

                HR = HR.cpu()
                im_h_y=np.squeeze(HR.data.numpy().astype(np.float32))
                im_h_y[im_h_y<0] = 0
                im_h_y[im_h_y>255.] = 255.
                image_hr = Image.fromarray(im_h_y.astype(np.uint8))
                image_hr.save(opt.output_path + image_hr_name[16:-17] + image_hr_name[-10:-4] + '_' + str(int(f)) +'.png','PNG')

print("Model=", opt.model)
print("It takes average {}s for processing".format(avg_elapsed_time/count))

# Find total parameters and trainable parameters
#total_params = sum(p.numel() for p in model.parameters())
#print(f'{total_params:,} total parameters.')
#total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f'{total_trainable_params:,} training parameters.')
