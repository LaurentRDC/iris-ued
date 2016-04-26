% Last Updated: 17/3/08
% Author: Christopher Galloway

% Wavelet Transform Script

% To perform a multilevel wavelet decomposition with db1 wavelet we can use:

[C,L]=wavedec(s,intLevels,strWavelet);

% where C contains all of the detail coeffecient and the latest approximate
% coefficients, L gives the lengths of each component (e.g cD1 is one
% component), and intLevels is the level approximation we want to go to.

% Threshold the Coefficients

% Hard Threshold the detail coefficients

for ii=1:intLevels;
    eval(sprintf('cD%d=detcoef(C,L,%d);',ii,ii));
    for jj=1:L(length(L)-ii);
        if eval(sprintf('abs(cD%d(%d))<=Thd(%d);',ii,jj,ii));
            eval(sprintf('cD%d(%d)=0;',ii,jj));
        end;
    end;
end;

% Soft Threshold the Approximate coefficients

eval(sprintf(strcat('cA%d=appcoef(C,L,''',strWavelet,''',%d);'),intLevels,intLevels))
for ii=1:L(1);
    if eval(sprintf('cA%d(%d)<=Tha;',intLevels,ii));
        eval(sprintf('cA%d(%d)=0;',intLevels,ii));
    else
        eval(sprintf('cA%d(%d)=cA%d(%d)-Tha;',intLevels,ii,intLevels,ii));
    end;
end;

% Reconstruct the coefficient vector C using the thresholded coefficients

strC=sprintf('C=[cA%d',intLevels);
for ii=1:intLevels;
    strC=strcat(strC,sprintf(' cD%d',intLevels+1-ii));
end;
strC=strcat(strC,'];');
eval(strC);


% Reconstruct the approximation and detail levels like so:

eval(sprintf(strcat('A%d=wrcoef(''a'',C,L,''',strWavelet,''',%d);'),intLevels,intLevels));
% eval(sprintf('assignin(''base'',''A%d'',A%d)',intLevels,intLevels));
for ii=1:intLevels;
    eval(sprintf(strcat('D%d=wrcoef(''d'',C,L,''',strWavelet,''',%d);'),ii,ii));
%     eval(sprintf('assignin(''base'',''D%d'',D%d)',ii,ii));
end;
