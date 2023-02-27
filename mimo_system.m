clc;
clf;
clear all;
close all;

%% VLSI DSP Final Project
iteration=10000;
SNR=[-5:1:20];
ZF_total=zeros(1,length(SNR));
MMSE_total=zeros(1,length(SNR));
ZF_BER=zeros(1,length(SNR));
MMSE_BER=zeros(1,length(SNR));
kbest2_BER=zeros(1,length(SNR));
kbest4_BER=zeros(1,length(SNR));
ML_BER=zeros(1,length(SNR));
tic
for n=1:length(SNR)
    %disp(['SNR:',num2str(n)]);
    ZF_error=0;
    MMSE_error=0;
    kbest2_error=0;
    kbest4_error=0;
    ML_error=0;
    for k=1:iteration
    % QPSK
    symbol_length=1024;
    symbol1=2*randi([0 1],2, symbol_length)-1;
    symbol2=2*randi([0 1],2, symbol_length)-1;
    s=[symbol1(1,:)+1j*symbol1(2,:);symbol2(1,:)+1j*symbol2(2,:)];

    
    
    %channel
    nt=2;
    nr=2;
    %H=randn(nr,nt)+1j*randn(nr,nt);
    H=(randn(nr,nt)+1j*randn(nr,nt))/sqrt(2);
    y=H*s;

    % noise
    SNR_Current=SNR(n);
    %baseband noise
    signal_power1=mean(abs(s(1,:)).^2);
    noise_power1=signal_power1*10^(-SNR_Current/10);
    signal_power2=mean(abs(s(2,:)).^2);
    noise_power2=signal_power2*10^(-SNR_Current/10);
    noise1=sqrt(noise_power1/2)*(randn(1,length(symbol1))+1j*randn(1,length(symbol1)));
    noise2=sqrt(noise_power2/2)*(randn(1,length(symbol2))+1j*randn(1,length(symbol2)));
       %passband noise
%     signal_power1=mean(abs(upconversion1).^2);
%     noise_power1=signal_power1*10^(-SNR_Current/10);
%     signal_power2=mean(abs(upconversion2).^2);
%     noise_power2=signal_power2*10^(-SNR_Current/10);
%     noise1=sqrt(noise_power1/2)*(randn(1,length(upconversion1))+1j*randn(1,length(upconversion1)));
%     noise2=sqrt(noise_power2/2)*(randn(1,length(upconversion2))+1j*randn(1,length(upconversion2)));

    y1=y(1,:)+noise1;
    y2=y(2,:)+noise2;
    b=[y1;y2];
   
    % ZF detector
    ZF=pinv(H)*b;

    %MMSE
    rho=[signal_power1/noise_power1, signal_power2/noise_power2];
    W=pinv(H'*H+diag(pinv(rho)))*H';
    MMSE=W*b;
    
    %kbest
    par.symbols=[1+j,1-j,-1+j,-1-j];
    par.nt=2;
    %k=4
    par.KBEST.K=4;
    k_best4=KBEST(H,b,par);
    %k=2
    par.KBEST.K=2;
    k_best2=KBEST(H,b,par);
    
    %ML
    ML_detected=ML(H,b,par);
    
    % calculate SNR
    %signal_power=mean(abs(x).^2);
    signal_power=mean(mean(abs(s).^2));
    %ZF_error_power=mean(abs(ZF-x).^2);
    ZF_error_power=mean(mean(abs(ZF-s).^2));
    %MMSE_error_power=mean(abs(MMSE-x).^2);
    MMSE_error_power=mean(mean(abs(MMSE-s).^2));

    %ZF_SNR=10*log10(signal_power/ZF_error_power);
    ZF_SNR=10*log10(signal_power/ZF_error_power);
    %MMSE_SNR=10*log10(signal_power/MMSE_error_power);
    MMSE_SNR=10*log10(signal_power/MMSE_error_power);
    %ZF_SNR=(signal_power/ZF_error_power);
    %MMSE_SNR=(signal_power/MMSE_error_power);
    ZF_total(n)=ZF_total(n)+ZF_SNR;
    MMSE_total(n)=MMSE_total(n)+MMSE_SNR;
    
    % detection
    detection=zeros(2,length(ZF));
    detection_MMSE=zeros(2,length(MMSE));

        for t=1:length(ZF)
           for idx=1:2
              if(real(ZF(idx,t))>=0)
                  detection(idx,t)=detection(idx,t)+1;
              elseif(real(ZF(idx,t))<0)
                  detection(idx,t)=detection(idx,t)-1;
              end

              if(imag(ZF(idx,t))>=0)
                  detection(idx,t)=detection(idx,t)+1j;
              elseif(imag(ZF(idx,t))<0)
                  detection(idx,t)=detection(idx,t)-1j;
              end
           end
        end
        
        for t=1:length(MMSE)
           for idx=1:2
              if(real(MMSE(idx,t))>=0)
                  detection_MMSE(idx,t)=detection_MMSE(idx,t)+1;
              elseif(real(MMSE(idx,t))<0)
                  detection_MMSE(idx,t)=detection_MMSE(idx,t)-1;
              end

              if(imag(MMSE(idx,t))>=0)
                  detection_MMSE(idx,t)=detection_MMSE(idx,t)+1j;
              elseif(imag(MMSE(idx,t))<0)
                  detection_MMSE(idx,t)=detection_MMSE(idx,t)-1j;
              end
           end
        end
        ZF_error=ZF_error+sum(sum(s~=detection));
        MMSE_error=MMSE_error+sum(sum(s~=detection_MMSE));
        kbest4_error=kbest4_error+sum(sum(s~=k_best4));
        kbest2_error=kbest2_error+sum(sum(s~=k_best2));
        ML_error=ML_error+sum(sum(s~=ML_detected));
        disp(['SNR:',num2str(n),'iteration:',num2str(k)]);
    end
    ZF_BER(n)=ZF_error/(2*iteration*symbol_length);
    MMSE_BER(n)=MMSE_error/(2*iteration*symbol_length);
    kbest2_BER(n)=kbest2_error/(2*iteration*symbol_length);
    kbest4_BER(n)=kbest4_error/(2*iteration*symbol_length);
    ML_BER(n)=ML_error/(2*iteration*symbol_length);
end
toc
% ZF_SNR=ZF_total/iteration;
% MMSE_SNR=MMSE_total/iteration;
ZF_performance=ZF_total/iteration;
MMSE_performance=MMSE_total/iteration;
% disp(['ZF SNR:', num2str(ZF_SNR)]);
% disp(['MMSE SNR:', num2str(MMSE_SNR)]);

figure(1);
plot(SNR,ZF_performance,'-*');
%semilogy(SNR,ZF_performance);
hold on;
plot(SNR,MMSE_performance,'-*');
%semilogy(SNR,MMSE_performance);
grid on;
legend('ZF','MMSE');
xlabel('SNR (dB)');
ylabel('SNRo (dB)');
title('ZF v.s MMSE');

figure(4);
semilogy(SNR,ML_BER,'-s');
hold on;
semilogy(SNR,ZF_BER,'-*');
%semilogy(SNR,ZF_performance);
hold on;
semilogy(SNR,MMSE_BER,'-*');
%semilogy(SNR,MMSE_performance);
hold on;
semilogy(SNR,kbest4_BER,'-o');
semilogy(SNR,kbest2_BER,'-o');
grid on;
legend('ML','ZF','MMSE','KBest, k=4','KBest, K=2');
xlabel('SNR (dB)');
ylabel('BER');
%title('ZF v.s MMSE');

figure(2);
subplot(2,2,1);
stem(real(s(1,:)));
hold on;
stem(real(ZF(1,:)));
legend('original','recovered');
title('real part of firsr input');

subplot(2,2,2)
stem(imag(s(1,:)));
hold on;
stem(imag(ZF(1,:)));
legend('original','recovered');
title('imag part of first input');

subplot(2,2,3);
stem(real(s(2,:)));
hold on;
stem(real(ZF(2,:)));
legend('original','recovered');
title('real part of second input');

subplot(2,2,4)
stem(imag(s(2,:)));
hold on;
stem(imag(ZF(2,:)));
legend('original','recovered');
title('imag part of second input');

figure(3);
subplot(2,2,1);
stem(real(s(1,:)));
hold on;
stem(real(MMSE(1,:)));
legend('original','recovered');
title('real part of firsr input');

subplot(2,2,2)
stem(imag(s(1,:)));
hold on;
stem(imag(MMSE(1,:)));
legend('original','recovered');
title('imag part of first input');

subplot(2,2,3);
stem(real(s(2,:)));
hold on;
stem(real(MMSE(2,:)));
legend('original','recovered');
title('real part of second input');

subplot(2,2,4)
stem(imag(s(2,:)));
hold on;
stem(imag(MMSE(2,:)));
legend('original','recovered');
title('imag part of second input');

% figure(3);
% plot(real(x(1,:)),imag(x(1,:)),'*');
% hold on;
% plot(real(ZF(1,:)),imag(ZF(1,:)),'*');
% legend('orignianl','ZF ');
% grid on;
% title('Constellation of the first signal');

%% Kbest detector
function detected=KBEST(H,y,par)
    %QR decomposition
    [Q,R]=qr(H);
    y_hat=Q'*y;
    detected=zeros(par.nt,length(y_hat));
    for n=1:length(y_hat)
    % -- Initialize Partial Euclidean Distance (PED) with last TX symbol
    PED_list=abs(par.symbols*R(par.nt,par.nt)-y_hat(par.nt,n));
    [PED_list,index]=sort(PED_list);
    s=par.symbols(:,index);
    
    %take Kbest
    s=s(:,1:min(par.KBEST.K,length(PED_list)));
    Kbest_PED_list=PED_list(1:min(par.KBEST.K,length(PED_list)));
    
    %for other Tx symbol (children of the root tree)
    for Layer=par.nt-1:-1:1
       PED_list=[];
       for k=1:length(Kbest_PED_list)
       tmp=Kbest_PED_list(k)+abs(par.symbols*R(Layer,Layer)-y_hat(Layer,n) + ...
          R(Layer,Layer+1:par.nt)*s(:,k)).^2;
       PED_list=[PED_list,tmp];    
       end
    %sorting
    s=[kron(ones(1,length(Kbest_PED_list)),par.symbols); ...
       kron(s,ones(1,length(par.symbols)))];
    [PED_list,idx]=sort(PED_list);
    s=s(:,idx);
     % take the K-best
    s=s(:,1:min(par.KBEST.K,length(PED_list)));
    Kbest_PED_list=PED_list(1:min(par.KBEST.K,length(PED_list)));
    end
   % -- take the best
  detected(:,n)=s(:,1);
    end

end

%% ML
function ML_detected=ML(H,y,par)

[height,width]=size(y);
detected=zeros(height,width);
x_hat=zeros(par.nt,length(par.symbols)^2);

for k=1:length(par.symbols)
    for t=1:length(par.symbols)
    x_hat(1,(k-1)*4+t)=par.symbols(k);
    x_hat(2,(k-1)*4+t)=par.symbols(t);
   end
end


    %disp(x_hat);
for k=1:length(y)

    tmp=sum(abs(y(:,k)-H*x_hat).^2);
    [~,idx]=min(tmp);
    
    detected(:,k)=x_hat(:,idx);

end
ML_detected=detected;
end

