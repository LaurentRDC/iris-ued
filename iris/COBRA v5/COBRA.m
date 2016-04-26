function varargout = GUI(varargin)
%GUI M-file for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('Property','Value',...) creates a new GUI using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to GUI_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      GUI('CALLBACK') and GUI('CALLBACK',hObject,...) call the
%      local function named CALLBACK in GUI.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 09-Oct-2008 15:28:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

datacursormode on

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in butPrevious.
function butPrevious_Callback(hObject, eventdata, handles)
% hObject    handle to butPrevious (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Decreases the index by 1 and plots the spectrum for the new index

if handles.index ~= 1;
    index=handles.index-1;
    handles.index=index;
    set(handles.txtSpectrumNo,'String',index);
    handles.spectra=handles.AllSpectra(index,:);
    handles.FilteredSpectra=handles.AllFilteredSpectra(index,:);
    handles.Background=handles.AllBackground(index,:);
    guidata(hObject,handles);
    PlotSpectra(handles);
    PlotBackground(handles);
    PlotFilteredSpectra(handles);
end;

% --- Executes on button press in butNext.
function butNext_Callback(hObject, eventdata, handles)
% hObject    handle to butNext (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Increases the index by 1 and plots the spectrum for the new index

if handles.index ~= handles.noSpectra;
    index=handles.index+1;
    handles.index=index;
    set(handles.txtSpectrumNo,'String',index);
    handles.spectra=handles.AllSpectra(index,:);
    handles.FilteredSpectra=handles.AllFilteredSpectra(index,:);
    handles.Background=handles.AllBackground(index,:);
    guidata(hObject,handles);
    PlotSpectra(handles);
    PlotBackground(handles);
    PlotFilteredSpectra(handles);
end;

% --- Executes on button press in butImport.
function butImport_Callback(hObject, eventdata, handles)
% hObject    handle to butImport (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Imports the data from a txt file where the first column is the x variable
% and every subsequent column is a new spectrum. The largest spectrum is
% then plotted

[filename,pathname] = uigetfile('*.txt','Open the signal file');
% mydata = readmulti(strcat(pathname,filename));
mydata=textread(strcat(pathname,filename));
% handles.wavenumber=mydata.wavenumbers';
% handles.AllSpectra=mydata.spectra';
handles.wavenumber=mydata(:,1)';
handles.AllSpectra=mydata(:,2:end)';
handles.noSpectra=size(handles.AllSpectra,1);
sumSpectra=sum(handles.AllSpectra,2);
[m,index]=max(sumSpectra);
handles.index=index;
set(handles.lblNoSpectra,'String',['of ',num2str(handles.noSpectra)]);
set(handles.txtSpectrumNo,'String',num2str(index));
handles.spectra=handles.AllSpectra(index,:);
handles.FilteredSpectra=handles.spectra;
handles.AllFilteredSpectra=handles.AllSpectra;
handles.length=length(handles.FilteredSpectra);
handles.Background=zeros(1,handles.length);
handles.BareBG=zeros(1,handles.length);
handles.adjBareBG=zeros(1,handles.length);
handles.AllBackground=zeros(handles.noSpectra,handles.length);
handles.ExcReg=[];
handles.noRegions=0;
set(handles.lstRegions,'Value',1);
for ii=1:15;
    eval(sprintf('set(handles.txtStart%d,''enable'',''off'')',ii));
    eval(sprintf('set(handles.txtFinish%d,''enable'',''off'')',ii));
end;

guidata(hObject,handles);
PlotSpectra(handles);
% isfield(mydata,'spectrum')
% handles = guidata(hObject)

function PlotSpectra(handles)

% Function to plot the unmodified spectrum

cla(handles.axes1,'reset');
cla(handles.axes2,'reset');
axes(handles.axes1)
plot(handles.wavenumber,handles.spectra);
title(handles.axes1,'Original Signal');
set(get(handles.axes1,'title'),'fontsize',12);
% xlabel(handles.axes1,'Wavenumbers');
% ylabel(handles.axes1,'Intensity');

function PlotBackground(handles)

% Function to plot the background on the same axes as the Original spectrum

hold on;
plot(handles.wavenumber,handles.Background);
title(handles.axes1,'Original Signal and Background');
set(get(handles.axes1,'title'),'fontsize',12);
% xlabel(handles.axes1,'Wavenumbers');
% ylabel(handles.axes1,'Intensity');

function PlotFilteredSpectra(handles)

% Function to plot the original spectrum minus the background

plot(handles.axes2,handles.wavenumber,handles.FilteredSpectra);
title(handles.axes2,'Filtered Signal');
set(get(handles.axes2,'title'),'fontsize',12);
% xlabel(handles.axes2,'Wavenumbers');
% ylabel(handles.axes2,'Intensity');


% --- Executes on button press in butInvFourier.
function butInvFourier_Callback(hObject, eventdata, handles)
% hObject    handle to butInvFourier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Performs a inverse fourier transform for after the FFT has been filtered

InvFourier(hObject,handles);
handles = guidata(hObject);
PlotSpectra(handles);
PlotBackground(handles);
PlotFilteredSpectra(handles);

function InvFourier(hObject,handles)

% The function that actually performs the inverse fourier transform 

handles.FilteredSpectra=real(ifft(handles.filFT));
guidata(hObject,handles);

% --- Executes on button press in butBackground.
function butBackground_Callback(hObject, eventdata, handles)
% hObject    handle to butBackground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Performs a wavelet transform background removal on the spectrum

WaveTrans(hObject,handles);
handles = guidata(hObject);
PlotSpectra(handles);
PlotBackground(handles);
PlotFilteredSpectra(handles);

function CreateExclusionRegion(hObject,handles)

% Define the region that is all background

ExcReg=[];
regions=handles.noRegions;
if regions>0
    for ii=1:regions;
        eval(sprintf('numStart=str2num(get(handles.txtStart%d,''String''));',ii));
        eval(sprintf('numFinish=str2num(get(handles.txtFinish%d,''String''));',ii));
         for jj=1:handles.length;
            if handles.wavenumber(jj)>=numStart && handles.wavenumber(jj)<=numFinish;
                ExcReg=[ExcReg,handles.wavenumber(jj)];
            end;
        end;
    end;
end;
handles.ExcReg=sort(ExcReg);
guidata(hObject,handles);

function WaveTrans(hObject,handles)

% The function that actually performs the wavelet transform with
% thresholding

list_entries = get(handles.lstWaveName,'String');
index_selected = get(handles.lstWaveName,'Value');
strWaveName=list_entries{index_selected(1)};
strWaveNo='';
if strcmp(get(handles.lstWaveNo,'Enable'),'on')==1;
    list_entries = get(handles.lstWaveNo,'String');
    index_selected = get(handles.lstWaveNo,'Value');
    strWaveNo=list_entries{index_selected(1)};
end;
strWavelet=strcat(strWaveName,strWaveNo);
intLevels=str2num(get(handles.txtLevels,'String'));
BGiter=str2num(get(handles.txtIterations,'String'));
Thd(1)=0.0;       
Thd(2)=0.0;
Thd(3)=0.0;
Thd(4)=0;
Thd(5)=0;
Thd(6)=0;
Thd(7)=0;
Thd(8)=0;
Thd(9)=0;
Thd(10)=0;
Tha=0;
Background=zeros(BGiter,handles.length);
s=handles.FilteredSpectra;

% Wavelet Transform Signal and set initial background as the approximate
% spectra

WaveletTransform;
eval(sprintf('Background(%d,:)=A%d;',1,intLevels));

% Correct Background signal that is offset due to raman modes and
% re-transform it BGiter times

    for ll=1:BGiter;
        for ii=1:handles.length;
            if handles.FilteredSpectra(ii)-Background(ll,ii)<5 || ismember(handles.wavenumber(ii),handles.ExcReg);
                Background(ll,ii)=handles.FilteredSpectra(ii);
            end;
        end;
        s=Background(ll,:);
%         assignin('base','newSignal',s);
        WaveletTransform;
        eval(sprintf('Background(%d,:)=A%d;',ll+1,intLevels));
    end;

% Remove final Background from the Original Signal and Perform Wavelet
% Transform

handles.Background=handles.Background+Background(BGiter+1,:);
handles.FilteredSpectra=handles.FilteredSpectra-Background(BGiter+1,:);
guidata(hObject,handles);


% --- Executes on button press in butFourier.
function butFourier_Callback(hObject, eventdata, handles)
% hObject    handle to butFourier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Perform the FFT on the filtered signal

Fourier(hObject,handles);
handles = guidata(hObject);
maxFT=max(abs(handles.FT));
t=1:handles.length;
cla(handles.axes7,'reset');
% cla(handles.axes2,'reset');
h=handles.axes7;
axes(h);
plot(t,abs(handles.FT));
hold on;
plot(t,maxFT*handles.fil);
title(handles.axes7,'Fourier Transform');
set(get(handles.axes1,'title'),'fontsize',12);
hold off
% title(handles.axes2,'Low Pass Filter');
% set(get(handles.axes2,'title'),'fontsize',12);

function Fourier(hObject,handles)

% The function that actually performs the fourier transform and filters it

t=1:handles.length;
width=str2num(get(handles.txtWidth,'String'));
cutoff=str2num(get(handles.txtCutoff,'String'));
handles.FT=fft(handles.FilteredSpectra);
fil=1-(1/4*(1-tanh((t-(handles.length/2+cutoff))/width)).*(1+tanh((t-(handles.length/2-cutoff))/width)));
handles.filFT=handles.FT.*fil;
handles.fil=fil;
guidata(hObject,handles);


function txtWidth_Callback(hObject, eventdata, handles)
% hObject    handle to txtWidth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtWidth as text
%        str2double(get(hObject,'String')) returns contents of txtWidth as a double


% --- Executes during object creation, after setting all properties.
function txtWidth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtWidth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtCutoff_Callback(hObject, eventdata, handles)
% hObject    handle to txtCutoff (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtCutoff as text
%        str2double(get(hObject,'String')) returns contents of txtCutoff as
%        a double


% --- Executes during object creation, after setting all properties.
function txtCutoff_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtCutoff (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtWavelet_Callback(hObject, eventdata, handles)
% hObject    handle to txtWavelet (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtWavelet as text
%        str2double(get(hObject,'String')) returns contents of txtWavelet as a double


% --- Executes during object creation, after setting all properties.
function txtWavelet_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtWavelet (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtLevels_Callback(hObject, eventdata, handles)
% hObject    handle to txtLevels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtLevels as text
%        str2double(get(hObject,'String')) returns contents of txtLevels as a double


% --- Executes during object creation, after setting all properties.
function txtLevels_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtLevels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtIterations_Callback(hObject, eventdata, handles)
% hObject    handle to txtIterations (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtIterations as text
%        str2double(get(hObject,'String')) returns contents of txtIterations as a double


% --- Executes during object creation, after setting all properties.
function txtIterations_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtIterations (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in butReset.
function butReset_Callback(hObject, eventdata, handles)
% hObject    handle to butReset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Resets all the data as if the user had only just imported it and defined
% the background region

handles.Background=handles.adjBareBG;
sumSpectra=sum(handles.AllSpectra,2);
[m,index]=max(sumSpectra);
handles.index=index;
handles.spectra=handles.AllSpectra(index,:);
handles.FilteredSpectra=handles.spectra-handles.Background;
handles.AllFilteredSpectra=handles.AllSpectra;
handles.AllBackground=zeros(handles.noSpectra,handles.length);
set(handles.txtSpectrumNo,'String',index)
guidata(hObject,handles);

PlotSpectra(handles);


% --- Executes on button press in butSave.
function butSave_Callback(hObject, eventdata, handles)
% hObject    handle to butSave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Saves the original spectra, background, wavenumber and filtered spectra
% to a .mat file

[filename,pathname]=uiputfile('*.mat','Save Workspace As:');
strFile=strcat(pathname,filename);
wavenumber=handles.wavenumber;
if handles.noSpectra==1;
    spectra=handles.spectra;
    background=handles.Background;
    filteredspectra=handles.FilteredSpectra;
else
    spectra=handles.AllSpectra;
    background=handles.AllBackground;
    filteredspectra=handles.AllFilteredSpectra; 
end;
save(strFile,'wavenumber','spectra','background','filteredspectra');


% --- Executes on button press in butFilterAll.
function butFilterAll_Callback(hObject, eventdata, handles)
% hObject    handle to butFilterAll (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Performs the background fit to all spectra

% profile on
noSpectra=handles.noSpectra;
CreateExclusionRegion(hObject,handles);
for ii=1:noSpectra;
    ii
    handles.spectra=handles.AllSpectra(ii,:);
    handles.FilteredSpectra=handles.AllSpectra(ii,:);
    handles.Background=zeros(1,handles.length);
%     if get(handles.radMinIntensity,'Value') == 1.0;
%         MinIntensity(hObject,handles);
%         handles = guidata(hObject);
%     end;
    if get(handles.radBareBG,'Value') == 1.0;
        BareBackground(hObject,handles);
        handles = guidata(hObject);
    end;
    if get(handles.radPolyBG,'Value') == 1.0;
        PolyBackground(hObject,handles);
        handles = guidata(hObject);
    end;
    if get(handles.radWavelet,'Value') == 1.0;
        WaveTrans(hObject,handles);
        handles = guidata(hObject);
    end;
    if get(handles.radFourier,'Value') == 1.0;
        Fourier(hObject,handles);
        handles = guidata(hObject);
        InvFourier(hObject,handles);
        handles = guidata(hObject);
    end;
    handles.AllFilteredSpectra(ii,:)=handles.FilteredSpectra;
    handles.AllBackground(ii,:)=handles.Background;
end;
index=1;
handles.index=index;
handles.spectra=handles.AllSpectra(index,:);
handles.FilteredSpectra=handles.AllFilteredSpectra(index,:);
handles.Background=handles.AllBackground(index,:);
set(handles.txtSpectrumNo,'String',index);
guidata(hObject,handles);
PlotSpectra(handles);
PlotBackground(handles);
PlotFilteredSpectra(handles);
% profile viewer
        
% --- Executes on button press in butPolyBackground.
function butPolyBackground_Callback(hObject, eventdata, handles)
% hObject    handle to butPolyBackground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Performs a polynomial fit to the spectrum

PolyBackground(hObject,handles);
handles = guidata(hObject);
PlotSpectra(handles);
PlotBackground(handles);
PlotFilteredSpectra(handles);

function PolyBackground(hObject,handles)

% Function that performs the polynomial fit

x=handles.wavenumber';
y=handles.FilteredSpectra';
% a0SP=mean(handles.FilteredSpectra);
noOrder=str2num(get(handles.txtPolyOrder,'String'));
% strSstart='s = fitoptions(''Method'',''NonlinearLeastSquares'',''Lower'',[-60000';
% strSmiddle='60000],''Startpoint'',[';
% strSend='a0SP';
% strPoly='a0';
% 
% for ii=1:noOrder;
%     strSstart=strcat(strSstart,',-60000');
%     strSmiddle=strcat('60000,',strSmiddle);
%     strSend=sprintf(strcat(strSend,' a%dSP'),ii);
%     eval(sprintf('a%dSP=0;',ii));
%     strPoly=sprintf(strcat(strPoly,'+a%d*x^%d'),ii,ii);
% end;
% 
% eval(strcat(strSstart,'],''Upper'',[',strSmiddle,strSend,']);'));
% eval(strcat('f = fittype(''',strPoly,''',''options'',s);'));

noIterations=str2num(get(handles.txtPolyIterations,'String'));
y=handles.FilteredSpectra';
for jj=1:noIterations
%     [c2,gof2]=fit(x,y,f);
    warning off;
    C=polyfit(x,y,noOrder);
    warning on

%     for ii=0:noOrder;
%         eval(sprintf('a%dPara=c2.a%d;',ii,ii));
%     end;

    for ii=0:noOrder;
        eval(sprintf('a%dPara=C(%d);',ii,noOrder+1-ii));
    end;

    strFit='y2(ii)=a0Para';
    for ii=1:noOrder;
        strFit=sprintf(strcat(strFit,'+a%dPara*x(ii)^%d'),ii,ii);
    end;

    for ii=1:handles.length;
        eval(strcat(strFit,';'));
        yDiff(ii)=y2(ii)-y(ii);
        if y2(ii)>handles.FilteredSpectra(ii)+5 || ismember(handles.wavenumber(ii),handles.ExcReg);
            y(ii)=handles.FilteredSpectra(ii);
        else
            y(ii)=y2(ii);
        end;
    end;

[maxDiff,index]=max(yDiff);
y2=y2-yDiff(index);

end;
handles.Background=handles.Background+y2;
handles.FilteredSpectra=handles.FilteredSpectra-y2;

guidata(hObject,handles);

% --- Executes on button press in butBareBackground.
function butBareBackground_Callback(hObject, eventdata, handles)
% hObject    handle to butBareBackground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Subtracts the bare background from the spectrum

BareBackground(hObject,handles);
handles = guidata(hObject);
PlotSpectra(handles);
PlotBackground(handles);
PlotFilteredSpectra(handles);

function BareBackground(hObject,handles)

% Function that actually removes the bare background

BareBG=handles.BareBG*str2num(get(handles.txtMultiplier,'String'))+str2num(get(handles.txtAddition,'String'));
handles.Background=BareBG;
handles.adjBareBG=BareBG;
handles.FilteredSpectra=handles.spectra-BareBG;
guidata(hObject,handles);


% --- Executes on button press in butImportBG.
function butImportBG_Callback(hObject, eventdata, handles)
% hObject    handle to butImportBG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Imports the bare background from a file containing only 1 spectrum. Must
% have the same x values as the original spectra

[filename,pathname] = uigetfile('*.txt','Open the Bare Background file');
mydata=textread(strcat(pathname,filename));
% mydata = readmulti(strcat(pathname,filename));
handles.Background=mydata(:,2)';
handles.BareBG=mydata(:,2)';
% handles.Background=mydata.spectra';
% handles.BareBG=mydata.spectra';
handles.adjBareBG=handles.BareBG;
handles.FilteredSpectra=handles.spectra-handles.BareBG;
guidata(hObject,handles);
PlotSpectra(handles);
PlotBackground(handles);
PlotFilteredSpectra(handles);


function txtMultiplier_Callback(hObject, eventdata, handles)
% hObject    handle to txtMultiplier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtMultiplier as text
%        str2double(get(hObject,'String')) returns contents of txtMultiplier as a double


% --- Executes during object creation, after setting all properties.
function txtMultiplier_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtMultiplier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtAddition_Callback(hObject, eventdata, handles)
% hObject    handle to txtAddition (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtAddition as text
%        str2double(get(hObject,'String')) returns contents of txtAddition as a double


% --- Executes during object creation, after setting all properties.
function txtAddition_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtAddition (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtPolyOrder_Callback(hObject, eventdata, handles)
% hObject    handle to txtPolyOrder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtPolyOrder as text
%        str2double(get(hObject,'String')) returns contents of txtPolyOrder as a double


% --- Executes during object creation, after setting all properties.
function txtPolyOrder_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtPolyOrder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtPolyIterations_Callback(hObject, eventdata, handles)
% hObject    handle to txtPolyIterations (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtPolyIterations as text
%        str2double(get(hObject,'String')) returns contents of txtPolyIterations as a double


% --- Executes during object creation, after setting all properties.
function txtPolyIterations_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtPolyIterations (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in radBareBG.
function radBareBG_Callback(hObject, eventdata, handles)
% hObject    handle to radBareBG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radBareBG


% --- Executes on button press in radPolyBG.
function radPolyBG_Callback(hObject, eventdata, handles)
% hObject    handle to radPolyBG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radPolyBG


% --- Executes on button press in radWavelet.
function radWavelet_Callback(hObject, eventdata, handles)
% hObject    handle to radWavelet (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radWavelet


% --- Executes on button press in radFourier.
function radFourier_Callback(hObject, eventdata, handles)
% hObject    handle to radFourier (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radFourier


function txtStart1_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart1 as text
%        str2double(get(hObject,'String')) returns contents of txtStart1 as a double


% --- Executes during object creation, after setting all properties.
function txtStart1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart2_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart2 as text
%        str2double(get(hObject,'String')) returns contents of txtStart2 as a double


% --- Executes during object creation, after setting all properties.
function txtStart2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart3_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart3 as text
%        str2double(get(hObject,'String')) returns contents of txtStart3 as a double


% --- Executes during object creation, after setting all properties.
function txtStart3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart4_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart4 as text
%        str2double(get(hObject,'String')) returns contents of txtStart4 as a double


% --- Executes during object creation, after setting all properties.
function txtStart4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart5_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart5 as text
%        str2double(get(hObject,'String')) returns contents of txtStart5 as a double


% --- Executes during object creation, after setting all properties.
function txtStart5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish1_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish1 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish1 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish2_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish1 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish2 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish3_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish1 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish3 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish4_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish4 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish4 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish5_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish5 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish5 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in butExclusionRegion.
function butExclusionRegion_Callback(hObject, eventdata, handles)
% hObject    handle to butExclusionRegion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CreateExclusionRegion(hObject,handles);
handles = guidata(hObject);
guidata(hObject,handles);


% --- Executes on selection change in lstRegions.
function lstRegions_Callback(hObject, eventdata, handles)
% hObject    handle to lstRegions (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns lstRegions contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lstRegions

% Enables the same number of background region txt boxes as is selected in
% lstRegions

list_entries = get(handles.lstRegions,'String');
index_selected = get(handles.lstRegions,'Value');
noRegions=list_entries{index_selected(1)};
noRegions=str2num(noRegions);
handles.noRegions=noRegions;

    for ii=1:15;
        if ii<=noRegions;
            eval(sprintf('set(handles.txtStart%d,''enable'',''on'')',ii));
            eval(sprintf('set(handles.txtFinish%d,''enable'',''on'')',ii));
        else
            eval(sprintf('set(handles.txtStart%d,''enable'',''off'')',ii));
            eval(sprintf('set(handles.txtFinish%d,''enable'',''off'')',ii));
        end;
    end;

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function lstRegions_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lstRegions (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in butResetBareBG.
function butResetBareBG_Callback(hObject, eventdata, handles)
% hObject    handle to butResetBareBG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Sets the bare background to zero

handles.FilteredSpectra=handles.spectra;
handles.Background=zeros(1,handles.length);
handles.BareBG=zeros(1,handles.length);
handles.adjBareBG=zeros(1,handles.length);
guidata(hObject,handles);

PlotSpectra(handles);


% --- Executes on button press in radMinIntensity.
function radMinIntensity_Callback(hObject, eventdata, handles)
% hObject    handle to radMinIntensity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radMinIntensity


% --- Executes on button press in butSaveBackground.
function butSaveBackground_Callback(hObject, eventdata, handles)
% hObject    handle to butSaveBackground (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in butSaveFilteredSpectra.
function butSaveFilteredSpectra_Callback(hObject, eventdata, handles)
% hObject    handle to butSaveFilteredSpectra (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in butImportOld.
function butImportOld_Callback(hObject, eventdata, handles)
% hObject    handle to butImportOld (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Import a .mat file which is a prevously background removed set of spectra

[filename,pathname] = uigetfile('*.mat','Open the matlab file');
mydata = load(strcat(pathname,filename));
handles.wavenumber=mydata.wavenumber;
handles.AllSpectra=mydata.spectra;
handles.noSpectra=size(handles.AllSpectra,1);
sumSpectra=sum(handles.AllSpectra,2);
[m,index]=max(sumSpectra);
handles.index=index;
set(handles.lblNoSpectra,'String',['of ',num2str(handles.noSpectra)]);
set(handles.txtSpectrumNo,'String',num2str(index));
handles.spectra=handles.AllSpectra(index,:);
handles.FilteredSpectra=mydata.filteredspectra(index,:);
handles.AllFilteredSpectra=mydata.filteredspectra;
handles.length=length(handles.FilteredSpectra);
handles.Background=mydata.background(index,:);
handles.BareBG=zeros(1,handles.length);
handles.adjBareBG=zeros(1,handles.length);
handles.AllBackground=mydata.background;
handles.ExcReg=[];
handles.noRegions=0;
set(handles.lstRegions,'Value',1);
for ii=1:15;
    eval(sprintf('set(handles.txtStart%d,''enable'',''off'')',ii));
    eval(sprintf('set(handles.txtFinish%d,''enable'',''off'')',ii));
end;

guidata(hObject,handles);
PlotSpectra(handles);
PlotBackground(handles);
PlotFilteredSpectra(handles);


function txtStart6_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart6 as text
%        str2double(get(hObject,'String')) returns contents of txtStart6 as a double


% --- Executes during object creation, after setting all properties.
function txtStart6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart7_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart7 as text
%        str2double(get(hObject,'String')) returns contents of txtStart7 as a double


% --- Executes during object creation, after setting all properties.
function txtStart7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart8_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart8 as text
%        str2double(get(hObject,'String')) returns contents of txtStart8 as a double


% --- Executes during object creation, after setting all properties.
function txtStart8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart9_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart9 as text
%        str2double(get(hObject,'String')) returns contents of txtStart9 as a double


% --- Executes during object creation, after setting all properties.
function txtStart9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart10_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart10 as text
%        str2double(get(hObject,'String')) returns contents of txtStart10 as a double


% --- Executes during object creation, after setting all properties.
function txtStart10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish6_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish6 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish6 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish7_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish7 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish7 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish8_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish8 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish8 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish9_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish9 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish9 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish10_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish10 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish10 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart11_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart11 as text
%        str2double(get(hObject,'String')) returns contents of txtStart11 as a double


% --- Executes during object creation, after setting all properties.
function txtStart11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart12_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart12 as text
%        str2double(get(hObject,'String')) returns contents of txtStart12 as a double


% --- Executes during object creation, after setting all properties.
function txtStart12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart13_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart13 as text
%        str2double(get(hObject,'String')) returns contents of txtStart13 as a double


% --- Executes during object creation, after setting all properties.
function txtStart13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart14_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart14 as text
%        str2double(get(hObject,'String')) returns contents of txtStart14 as a double


% --- Executes during object creation, after setting all properties.
function txtStart14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtStart15_Callback(hObject, eventdata, handles)
% hObject    handle to txtStart15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtStart15 as text
%        str2double(get(hObject,'String')) returns contents of txtStart15 as a double


% --- Executes during object creation, after setting all properties.
function txtStart15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtStart15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish11_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish11 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish11 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish12_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish12 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish12 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish13_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish13 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish13 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish14_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish14 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish14 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function txtFinish15_Callback(hObject, eventdata, handles)
% hObject    handle to txtFinish15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txtFinish15 as text
%        str2double(get(hObject,'String')) returns contents of txtFinish15 as a double


% --- Executes during object creation, after setting all properties.
function txtFinish15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtFinish15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in lstWaveName.
function lstWaveName_Callback(hObject, eventdata, handles)
% hObject    handle to lstWaveName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns lstWaveName contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lstWaveName

% Changes the values in lstWaveNo depending on which wavelet name is
% choosen and then plots the wavelet

list_entries = get(handles.lstWaveName,'String');
index_selected = get(handles.lstWaveName,'Value');
set(handles.lstWaveNo,'Value',1);
strWaveName=list_entries{index_selected(1)};
if strcmp(strWaveName,'haar')==1;
    set(handles.lstWaveNo,'Enable','off');
    strWave='haar';
end;
if strcmp(strWaveName,'db')==1;
    set(handles.lstWaveNo,'Enable','on');
    set(handles.lstWaveNo,'String',{'1','2','3','4','5','6','7','8','9','10'});
    strWave='db1';
end;
if strcmp(strWaveName,'sym')==1;
    set(handles.lstWaveNo,'Enable','on');
    set(handles.lstWaveNo,'String',{'2','3','4','5','6','7','8'});
    strWave='sym2';
end;
if strcmp(strWaveName,'coif')==1;
    set(handles.lstWaveNo,'Enable','on');
    set(handles.lstWaveNo,'String',{'1','2','3','4','5'});
    strWave='coif1';
end;
if strcmp(strWaveName,'bior')==1;
    set(handles.lstWaveNo,'Enable','on');
    set(handles.lstWaveNo,'String',{'1.1','1.3','1.5','2.2','2.4','2.6','2.8','3.1','3.3','3.5','3.7','3.9','4.4','5.5','6.8'});
    strWave='bior1.1';
end;
if strcmp(strWaveName,'rbio')==1;
    set(handles.lstWaveNo,'Enable','on');
    set(handles.lstWaveNo,'String',{'1.1','1.3','1.5','2.2','2.4','2.6','2.8','3.1','3.3','3.5','3.7','3.9','4.4','5.5','6.8'});
    strWave='rbio1.1';
end;
if strcmp(strWaveName,'dmey')==1;
    set(handles.lstWaveNo,'Enable','off');
    strWave='dmey';
end;

[PHI,PSI,XVAL] = WAVEFUN(strWave,4);
cla(handles.axes4,'reset');
h=handles.axes4;
axes(h);
plot(PSI);
maxPSI=max(abs(PSI));
set(handles.axes4,'XLim',[0 length(PSI)]);
set(handles.axes4,'Ylim',[-maxPSI-0.1 maxPSI+0.1]);

handles = guidata(hObject);

% --- Executes during object creation, after setting all properties.
function lstWaveName_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lstWaveName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in lstWaveNo.
function lstWaveNo_Callback(hObject, eventdata, handles)
% hObject    handle to lstWaveNo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns lstWaveNo contents as cell array
%        contents{get(hObject,'Value')} returns selected item from
%        lstWaveNo

% Changes the wavelet which will be used in the decomposition and plots the
% new wavelet

list_entries = get(handles.lstWaveName,'String');
index_selected = get(handles.lstWaveName,'Value');
strWaveName=list_entries{index_selected(1)};
list_entries2 = get(handles.lstWaveNo,'String');
index_selected2 = get(handles.lstWaveNo,'Value');
strWaveNo=list_entries2{index_selected2(1)};
% ischar(list_entries)
strWave=strcat(strWaveName,strWaveNo);

[PHI,PSI,XVAL] = WAVEFUN(strWave,4);
cla(handles.axes4,'reset');
h=handles.axes4;
axes(h);
plot(PSI);
maxPSI=max(abs(PSI));
set(handles.axes4,'XLim',[0 length(PSI)]);
set(handles.axes4,'Ylim',[-maxPSI-0.1 maxPSI+0.1]);

handles = guidata(hObject);


% --- Executes during object creation, after setting all properties.
function lstWaveNo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lstWaveNo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in butSaveExcRegion.
function butSaveExcRegion_Callback(hObject, eventdata, handles)
% hObject    handle to butSaveExcRegion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Saves the defined background region in a .mat file

list_entries = get(handles.lstRegions,'String');
index_selected = get(handles.lstRegions,'Value');
noRegions=str2num(list_entries{index_selected(1)});
for ii = 1:noRegions;
    eval(sprintf('start(%d)=str2num(get(handles.txtStart%d,''String''));',ii,ii));
    eval(sprintf('finish(%d)=str2num(get(handles.txtFinish%d,''String''));',ii,ii));
end;
[filename,pathname]=uiputfile('*.mat','Save File As:');
strFile=strcat(pathname,filename);
save(strFile,'start','finish');

% --- Executes on button press in butImportExcRegion.
function butImportExcRegion_Callback(hObject, eventdata, handles)
% hObject    handle to butImportExcRegion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Imports a background region previously defined and fills the values into txt boxes 

[filename,pathname]=uigetfile('*.mat','Choose file:');
strFile=strcat(pathname,filename);
load(strFile);
noRegions=length(start);
set(handles.lstRegions,'Value',noRegions+1);
for ii=1:15;
    if ii<=noRegions;
        eval(sprintf('set(handles.txtStart%d,''String'',start(%d));',ii,ii));
        eval(sprintf('set(handles.txtFinish%d,''String'',finish(%d));',ii,ii));
        eval(sprintf('set(handles.txtStart%d,''Enable'',''on'');',ii));
        eval(sprintf('set(handles.txtFinish%d,''Enable'',''on'');',ii));
    else
        eval(sprintf('set(handles.txtStart%d,''String'',0);',ii,ii));
        eval(sprintf('set(handles.txtFinish%d,''String'',0);',ii,ii));
        eval(sprintf('set(handles.txtStart%d,''Enable'',''off'');',ii));
        eval(sprintf('set(handles.txtFinish%d,''Enable'',''off'');',ii));
    end;
end;



function txtSpectrumNo_Callback(hObject, eventdata, handles)
% hObject    handle to txtSpectrumNo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% When the user presses enter, the choosen spectrum is displayed and set as
% the index

% k = get(gcf, 'CurrentKey')
% % strcmp(k,'return')
% % if strcmp(k,'return')==1;
    strIndex=get(handles.txtSpectrumNo,'String');
    index=str2num(strIndex);
    handles.index=index;
    handles.spectra=handles.AllSpectra(index,:);
    handles.FilteredSpectra=handles.AllFilteredSpectra(index,:);
    handles.Background=handles.AllBackground(index,:);
    guidata(hObject,handles);
    PlotSpectra(handles);
    PlotBackground(handles);
    PlotFilteredSpectra(handles); 
% end;

% Hints: get(hObject,'String') returns contents of txtSpectrumNo as text
%        str2double(get(hObject,'String')) returns contents of txtSpectrumNo as a double


% --- Executes during object creation, after setting all properties.
function txtSpectrumNo_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txtSpectrumNo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in butResetSingle.
function butResetSingle_Callback(hObject, eventdata, handles)
% hObject    handle to butResetSingle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Resets the background fit and filtering of the spectrum being worked on

handles.Background=handles.adjBareBG;
index=handles.index;
handles.spectra=handles.AllSpectra(index,:);
handles.FilteredSpectra=handles.spectra-handles.Background;
handles.AllFilteredSpectra=handles.AllSpectra;
handles.AllBackground=zeros(handles.noSpectra,handles.length);
guidata(hObject,handles);

PlotSpectra(handles);

% --- Executes on button press in butSaveSingle.
function butSaveSingle_Callback(hObject, eventdata, handles)
% hObject    handle to butSaveSingle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Saves just the current indexed spectrum and filtering

[filename,pathname]=uiputfile('*.mat','Save Workspace As:');
strFile=strcat(pathname,filename);
wavenumber=handles.wavenumber;
handles.noSpectra==1;
spectra=handles.spectra;
background=handles.Background;
filteredspectra=handles.FilteredSpectra;

save(strFile,'wavenumber','spectra','background','filteredspectra');



