% load blur image
video_name='camerashake';
load('../data_example/data.mat');
t_shift=-0.04*1e6;

fid = fopen('../data_example/test.txt','w');
for index_ii=2:length(matlabdata.data.frame.samples)-1
    % find events during take the photo
    fprintf(fid,'%s ',['../data_example/'  video_name '_blurimage/' num2str(index_ii,'%04d') '.png']);

    t_for=matlabdata.data.frame.timeStampStart(index_ii+1)-matlabdata.data.frame.timeStampEnd(index_ii);
    t_back=matlabdata.data.frame.timeStampStart(index_ii)-matlabdata.data.frame.timeStampEnd(index_ii-1);
    startnum=matlabdata.data.frame.timeStampStart(index_ii)+t_shift-t_back/2;
    endnum=matlabdata.data.frame.timeStampEnd(index_ii)+t_shift+t_for/2;

    event_stream=[];

    event_index=find(matlabdata.data.polarity.timeStamp>=startnum & matlabdata.data.polarity.timeStamp<=endnum);

    num=0;

    for i=event_index(1):event_index(end)
        num=num+1;
        event_stream(num,1) = matlabdata.data.polarity.timeStamp(i);
        event_stream(num,3) = matlabdata.data.polarity.x(i);
        event_stream(num,2) = matlabdata.data.polarity.y(i);
        event_stream(num,4) = matlabdata.data.polarity.polarity(i);
    end

    % event to frame
    timeduring=double(endnum-startnum);
    event_stream(:,1)=(event_stream(:,1)-double(startnum))/timeduring;
    dlmwrite(['../data_example/'  video_name '_event_frame_txt/' num2str(index_ii,'%04d') '.txt'],event_stream,'delimiter',' ','precision',8,'newline','pc');
    
    fprintf(fid,'%s\n',['../data_example/'  video_name '_event_frame_txt/' num2str(index_ii,'%04d') '.txt']);
end

fclose(fid);
