using WAV



function ar1Filter(x,Fs,fq,decayTime)
  a=exp((2.0*im*pi*fq - 1/decayTime)/Fs)
  normalizationFactor = 1-abs(a)
  y=zeros(Complex64,length(x))
  for it in 1:length(x)
    y[it]=a*get(y,it-1,0.0)+normalizationFactor*x[it]
  end
  return y
end



#fqA4 = 440.0
nFq = 30
fMin=55.0
fMax=1E4
filteringFq=exp(collect(linspace(log(fMin),log(fMax),nFq)))
decayTime =  4/fMin*filteringFq/filteringFq[1]
timeToWait=0.1 # (seconds) before next attack
timeMax=0.1 # (seconds) duration of the segment from which the features are extracted
nSampleMax=round(Int,timeMax*Fs)


function featureExtract(x,Fs)
  a=Array(Float64,nFq)
  for iFq in 1:nFq
    v=abs(ar1Filter(x,Fs,filteringFq[iFq],decayTime[iFq]))
    timeMaxInt=minimum([ceil(Int,timeMax*Fs),length(v)])
    a[iFq]=mean(v[1:timeMaxInt])
  end
  a=a/norm(a)
  return a
end


function attackDetect(x,Fs)
  myEps=1E-2
  thr=2.0
  nSampleToWait=round(Int,timeToWait*Fs)
  idxOfAttackSet=Set{Int64}()
  y=ar1Filter(abs(x),Fs,0,1.0)
  logDy=zeros(Float64,length(y))
  nSampleFromPreviousAttack=Inf
  for it in 2:length(y)
    logDy[it]=(log(y[it]+myEps)-log(y[it-1]+myEps))*Fs
    if(logDy[it]>thr && nSampleFromPreviousAttack>nSampleToWait)
      nSampleFromPreviousAttack=0
      push!(idxOfAttackSet,it)
    else
      nSampleFromPreviousAttack+=1
    end
  end
  return (logDy,idxOfAttackSet)
end


sourceSampleIdDict=Dict{Int,Any}()
destSampleIdDict=Dict{Int,Any}()
sourceSampleIdToFeat=Dict{Int,Any}()


sourceFilenameToDestFilename=Dict{ASCIIString,ASCIIString}(
  "mouthSnare.wav" => "snare2.wav",
  "mouthSnare2.wav" => "snare2.wav",
  "mouthSnare3.wav" => "snare2.wav",
  "mouthHihat.wav" => "hihat2.wav",
  "mouthHihat2.wav" => "hihat2.wav",
  "mouthHihat3.wav" => "hihat2.wav",
  "mouthRim.wav" => "rimShot.wav",
  "mouthRim2.wav" => "rimShot.wav",
  "mouthCrash.wav" => "crash2.wav",
  "mouthBassDrum2.wav" => "bassDrum2.wav",
  "mouthBassDrum.wav" => "bassDrum2.wav")


sampleId=1
for sourceFilename in keys(sourceFilenameToDestFilename)
  destFilename = sourceFilenameToDestFilename[sourceFilename]
  sourceSample,Fs=wavread(sourceFilename)
  sourceSample=sourceSample[:,1]
  destSample,Fs=wavread(destFilename)
  destSample=destSample[:,1]
  sourceSampleDict[sampleId]=sourceSample
  destSampleDict[sampleId]=destSample
  (temp,idxSet)=attackDetect(sourceSample,Fs)
  idx1=sort(collect(idxSet))[1]
  #idx1=1
  phi=featureExtract(get(sourceSample,idx1:idx1+nSampleMax,0.0),Fs)
  sourceSampleIdToFeat[sampleId]=phi
  sampleId+=1
end


function addAt!(toMod,toAdd,idx)
  idxEnd=minimum([length(toMod),idx+length(toAdd)-1])
  len=length(toMod[idx:idxEnd])
  toMod[idx:idxEnd]+=toAdd[1:len]
end


function transform()
  x,Fs=wavread("in.wav")
  (logDy,idxOfAttackSet)=attackDetect(x,Fs)
  y=zeros(Float64,length(x))
  for idxTime in idxOfAttackSet
    phi=featureExtract(get(x,idxTime:idxTime+nSampleMax,0.0),Fs)
    distTo=Array(Float64,length(sourceFilenameToDestFilename))
    for sampleId in keys(sourceSampleIdToFeat)
      #distTo[sampleId]=dot(sourceSampleIdToFeat[sampleId],phi)
      distTo[sampleId]=log(norm(sourceSampleIdToFeat[sampleId]-phi))
    end
    (tempVal,idxMin)=findmin(distTo)
    println(idxMin)
    #println(idxTime)
    addAt!(y,destSampleDict[idxMin],idxTime)
  end
  return y
end


y=transform()


z=y/maximum(abs(y))
#myPlot(collect(1:length(z)),z)
wavwrite(z,"out.wav",Fs=Fs)









