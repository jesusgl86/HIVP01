N=18;
% for i = 1:N 
%    filename = [num2str(i) '.tif'];
%    t = Tiff(filename,'r'); 
%    img = readRGBAImage(t);
%    XTrain0(:,:,:,i) =img;
%    % do something with img
% end
imageSize=[1024 1024 3]
imageSizeTensor=[1024 1024 3 18]
customreader=@(x) readRGBAImage(convertCharsToStrings(Tiff(x,'r')));
loc_name="C:\Users\viu837\OneDrive - The University of Texas-Rio Grande Valley\Documents\MATLAB\HIV\STAT1";
imds = imageDatastore(loc_name,"FileExtensions",".tif","ReadFcn",customreader);
imgs=readall(imds);
XTrain0 =reshape(cat(3,imgs{:}),imageSizeTensor);

%%
theta=15;
Angle=theta:theta:360-theta;
M=length(Angle);
for i = 1:M
    for j=1:N
        XTrain0(:,:,:,N*i+j) = imrotate(XTrain0(:,:,:,j),Angle(i),'bicubic','crop');
    end
    i
end
y=xlsread("y.xlsx");
y=repmat(y,(M+1),1);

%%
Xsplit=round(N*(M+1)*.8);
if theta==30
    Numvalid=20;
elseif theta==15
    Numvalid=40;    
elseif theta==10 
    Numvalid=60;
elseif theta==5
    Numvalid=120;
elseif theta==1
    Numvalid=600;
end
%validation set size 20-30, 40-15, 60-10, 120-5
idx=randperm(N*(M+1),Xsplit);
valididx0=randperm(numel(idx),Numvalid); %
Trainidx=idx(~ismember(1:Xsplit,valididx0));
Valididx=idx(ismember(1:Xsplit,valididx0));
Trainid=ismember(1:N*(M+1),Trainidx);
Validid=ismember(1:N*(M+1),Valididx);
Testid=~ismember(1:N*(M+1),idx);
XTrain=XTrain0(:,:,:,Trainid);
Xvalid=XTrain0(:,:,:,Validid);
Xtest=XTrain0(:,:,:,Testid);
Toutput=y(Trainid);
Toutvalid=y(Validid);
Touttest=y(Testid);

% theta=10;
% Angle=theta:theta:360-theta;
% M=length(Angle);
% Nv=size(Toutvalid,1);
% Nt=size(Touttest,1);
% for i = 1:M
%     for j=1:Nv
%         Xvalid(:,:,:,Nv*i+j) = imrotate(Xvalid(:,:,:,j),Angle(i),'bicubic','crop');
%     end
%     for j=1:Nt
%         Xtest(:,:,:,Nt*i+j) = imrotate(Xtest(:,:,:,j),Angle(i),'bicubic','crop');
%     end
%     i
% end


% Toutvalid=repmat(Toutvalid,(M+1),1);
% Touttest=repmat(Touttest,(M+1),1);
%%
Toutput=categorical(Toutput);
Toutvalid=categorical(Toutvalid);
Touttest=categorical(Touttest);
classNames=categories(Toutput);
numClasses=numel(classNames);
imageAugmenter = imageDataAugmenter( 'RandRotation',[0,360])

augimds = augmentedImageDatastore(imageSize,XTrain,Toutput,'DataAugmentation',imageAugmenter);







%% Training
layers = [
    imageInputLayer([1024 1024 3])
    convolution2dLayer(3,8)%,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer(Name="softmax")
    classificationLayer];

miniBatchSize  = 12;
validationFrequency = floor(numel(Toutput)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','cpu',... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.99, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xvalid,Toutvalid}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(augimds,layers,options);

%% Testing
YPredtest = classify(net,Xtest);

Testaccuracy = sum(YPredtest == Touttest)/numel(Touttest)*100

sum(YPredtest == Touttest)
numel(Touttest)

Class0Correct=sum(YPredtest(YPredtest==categorical(0))==Touttest(Touttest==categorical(0)))
Class0Total=sum(YPredtest==categorical(0))
TestaccuracyClass0=Class0Correct/Class0Total*100

Class1Correct=sum(YPredtest(YPredtest==categorical(1))==Touttest(Touttest==categorical(1)))
Class1Total=sum(YPredtest==categorical(1))
TestaccuracyClass1=Class1Correct/Class1Total*100

Class2Correct=sum(YPredtest(YPredtest==categorical(2))==Touttest(Touttest==categorical(2)))
Class2Total=sum(YPredtest==categorical(2))
TestaccuracyClass2=Class2Correct/Class2Total*100

Class3Correct=sum(YPredtest(YPredtest==categorical(3))==Touttest(Touttest==categorical(3)))
Class3Total=sum(YPredtest==categorical(3))
TestaccuracyClass3=Class3Correct/Class3Total*100




