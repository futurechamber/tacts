import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from joblib import Parallel, delayed
import pyunicorn.timeseries.recurrence_plot as rp
from collections import defaultdict
import numpy as np
import math
import random

# TRANSFORMATION COST TIME-SERIES V4.3
# Çelik Özdeş 2021 (but some code is borrowed)

#  V4+ FEATURES:
# - Overlapping windows
# - Multivariate TS support
# - Sampling Rate normalization
# - Non-biased Unit Cost fitting
# - Support for Gap Masking
# - Parallel Lambda Optimization
# - Parallel Spectrum Pipeline
# - Surrogate Bootsrapping

# TACTS CORE METHODS

def Generate(F, small_win, p2=1, T0=None, TN=None, dt=None, report=0, figsize=(12,5)):

    Eps = 1e-10
    N, Dim = F.shape[1], F.shape[0]-1

    # Set Timeline
    windowsize = small_win*np.mean(np.diff(F[0]))
    if dt is None: dt = windowsize
    if T0 is None: T0 = min(F[0]) + windowsize
    if TN is None: TN = max(F[0]) - windowsize

    if dt > windowsize : print("Warning : Interval not spanned.")
    if (T0 - windowsize) < min(F[0]) or TN+windowsize>max(F[0]):
        print('Warning: Bad Timeline. T:', T0,TN,dt,'Data:', min(F[0]),max(F[0]),'W:',windowsize,'(',small_win,')')

    timecenter = np.arange(T0,  TN+Eps, dt)
    windows_left = timecenter - windowsize
    windows_right = timecenter + windowsize
    N_win = len(timecenter)

    # Tune p0 (unit shift)
    tdifs = np.diff(F[0])
    p0 = 1/np.mean(tdifs[tdifs < windowsize])
    p0_old = (len(F[0]-1)/(F[0,-1]-F[0,0]))

    # Tune p1 (unit scale)
    difs = []
    for i in range(N_win):
        t1 = np.where( (F[0] >= windows_left[i]) & (F[0] <= timecenter[i]) )[0]
        t2 = np.where( (F[0] >= timecenter[i]) & (F[0] <= windows_right[i]) )[0]
        if len(t1) and len(t2):
            difs.append([abs(np.mean(F[k+1,t1])-np.mean(F[k+1,t2])) for k in range(Dim)])
    p1 = 1 / np.mean(difs)

    # Calculate TACTS
    values, MinInserts, SN = TACTS(F[0], F[1:],
        windows_left, timecenter, windows_right, p0, p1, p2, output = report)

    # Generate Masks
    gaps = np.where(values < 0)[0]
    mask = np.ones(len(values), dtype=bool)
    mask[gaps] = False
    values[gaps] = 0

    if report:
        print("Dimensions :",Dim, "\nDatapoints :",N)
        print("Window Size :",windowsize)
        print("Tacts Length :",N_win)
        print("Shift/Scale Unit Costs > >",p0,"/", p1)
        print("Ignore Cost > >",p2)
        if report > 1:
            fig = plt.figure(figsize=figsize)
            plt.xlim(min(F[0])-Eps,max(F[0])+Eps)
            #for w in timecenter: plt.axvline(w, alpha=0.3,lw=0.5)
            plt.plot(timecenter, values, '-ok', label='Tacts', ms=2, lw=1)
            if report > 2:
                fig = plt.figure(figsize=figsize)
                plt.xlim(min(F[0])-Eps,max(F[0])+Eps)
                plt.plot(timecenter,SN,'.r', ms=2)

    # Format
    TC = np.empty(shape=(2,len(values)))
    TC[0,:] = timecenter
    TC[1,:] = values
    return TC, gaps, mask

def TACTS(t, x, winleft, timecenter, winright, p0, p1, p2, output=False):
    SIZE = len(timecenter)
    dist = np.zeros(SIZE)
    winshifts, winadds = 0, 0
    AVGSHIFT, AVGSCALE = 0, 0
    ForcedInserts, Inserts, SN = 0, [], []
    AvgSkipRatio = 0
    for i in range(SIZE):

        # Arrange Time Sections
        t1 = np.where((t >= winleft[i]) & (t < timecenter[i]))[0]
        t2 = np.where((t >= timecenter[i]) & (t < winright[i]))[0]

        ti = t[t1] - winleft[i]
        tj = t[t2] - timecenter[i]

        INSERTS = len(t2)-len(t1)
        Inserts.append(INSERTS)
        ForcedInserts += abs(INSERTS)
        SN.append(len(t1))

        # Calculate Distances
        dist[i], shifts, adds, avgshiftcost, avgscalecost, skipratio = spikedist(ti, tj, x[:,t1], x[:,t2], p0, p1, p2)

        # Collect Statistics
        AVGSHIFT += avgshiftcost
        AVGSCALE += avgscalecost
        winshifts += shifts
        winadds += adds
        AvgSkipRatio += skipratio


    AVGSHIFT /= SIZE
    AVGSCALE /= SIZE
    AvgSkipRatio /= SIZE

    if output:
        print('Points in Segments : [%d,%d]'%(min(SN),max(SN)))
        print('Forced Skips : %d Total (%d,%d)'%(ForcedInserts, min(Inserts),max(Inserts)))
        print('All Partial Paths > Shift : %d  Insert : %d'%(winshifts,winadds))
        print('Average Skip Ratio :',AvgSkipRatio)
        print('Avg. Shift Cost :', AVGSHIFT)
        print('Avg. Scale Cost :', AVGSCALE)

    return dist, Inserts, SN

def spikedist(ti, tj, xi, xj, p0, p1, p2, euclidian=False, stats=True):
    Dim = xi.shape[0]

    # Initialize Transformation Costs Matrix
    nspi = len(ti); nspj = len(tj)
    scr = np.zeros([nspi+1,nspj+1])
    scr[:,0] = p2*np.transpose(np.arange(nspi+1))
    scr[0,:] = p2*np.arange(nspj+1)
    if stats:
        stat = np.zeros([nspi+1,nspj+1])
        stat[:,0] = np.transpose(np.arange(nspi+1))
        stat[0,:] = np.arange(nspj+1)

    shifts, adds = 0,0
    avgshiftcost, avgscalecost = 0,0

    if (nspi>0) and (nspj>0):

        # Dynamic Programming
        for i in range(1,nspi+1):
            for j in range(1,nspj+1):
                timedif = abs(ti[i-1]-tj[j-1])
                ampdif = 0
                for k in range(Dim):
                    ampdif += (xi[k,i-1]-xj[k,j-1])**2
                scalecost = p1 * (ampdif **(0.5))
                shiftcost = p0 * timedif
                avgscalecost += scalecost
                avgshiftcost += shiftcost
                if euclidian:
                    cost = ((shiftcost**2) + (scalecost**2))**(0.5)
                else :
                    cost = shiftcost + scalecost

                if scr[i-1,j-1]+cost < min(scr[i-1,j]+p2, scr[i,j-1]+p2):
                    shifts+=1
                    stat[i,j]=stat[i-1,j-1]
                elif scr[i-1,j]<scr[i,j-1]:
                    adds+=1
                    stat[i,j]=stat[i-1,j]+1
                else:
                    adds+=1
                    stat[i,j]=stat[i,j-1]+1

                scr[i,j]=min(scr[i-1,j]+p2, scr[i,j-1]+p2, scr[i-1,j-1]+cost)

        d = scr[-1,-1]
        skipratio = stat[-1,-1]/(nspi+nspj)
        avgscalecost /= (nspi*nspj)
        avgshiftcost /= (nspi*nspj)

    else:
        d = -1
        skipratio = 1

    if (nspi+nspj)!=0:
        d=d/(nspi+nspj)
    else: d = -1

    return d, shifts, adds, avgshiftcost, avgscalecost, skipratio

def GenerateSpectrum(F, windowsizes, OPT, **kwargs):
    print('Generating TACTS (%d) '%len(windowsizes), end='')
    S = []
    for w in range(len(windowsizes)):
        print('.',end='')
        S.append(Generate(F, windowsizes[w], p2=OPT[w][2], **kwargs)[0])
    print('Done.')
    return S

# OPTIMIZATION METHODS

def OptimizeWindow(F, winsize, T0=None, TN=None, dt=None, Lmin=0.1, Lmax=20, ResZero=20, Focus=0.5, ResDepth=3, report=2, NORM_CDF=None):

    if report: print("Optimizing W%.2f"%winsize," >> Res %d / Focus %.2f / Depth %d"%(ResZero,Focus,ResDepth), flush=True)
    if NORM_CDF is None : NORM_CDF = normalCDF()
    LAMBDAS, KSDS, STDS, MAXS = [],[],[],[]
    DL = (Lmax-Lmin)/(ResZero-1)
    for L in np.linspace(Lmin,Lmax,ResZero):
        LAMBDAS.append(L)
        tc, gaps, mask = Generate(F, winsize, L, T0=T0, TN=TN, dt=dt)
        STDS.append(np.std(tc[1][mask]))
        MAXS.append(np.max(tc[1][:]))
        TACTS_CDF = CDF(tc[1][mask])
        KSDS.append(max(abs(TACTS_CDF - NORM_CDF)))

    if report>1:
        plt.figure(figsize=(12,3))
        plt.plot(LAMBDAS,KSDS,'ok',ms=2)

    for depth in range(1,ResDepth+1):
        DL /= 3
        minIndexes = sorted(range(len(KSDS)), key=lambda k: KSDS[k])
        FocalPoints = int(Focus*len(LAMBDAS))
        for k in minIndexes[:FocalPoints]:
            if LAMBDAS[k]+DL < Lmax:
                LAMBDAS.append(LAMBDAS[k]+DL)
                tc, gaps, mask = Generate(F, winsize, LAMBDAS[k]+DL, T0=T0, TN=TN, dt=dt)
                STDS.append(np.std(tc[1][mask]))
                MAXS.append(np.max(tc[1][:]))
                TACTS_CDF = CDF(tc[1][mask])
                KSDS.append(max(abs(TACTS_CDF - NORM_CDF)))
                if report>1: plt.plot(LAMBDAS[-1],KSDS[-1],'ob',ms=2)
            if LAMBDAS[k]-DL > Lmin:
                LAMBDAS.append(LAMBDAS[k]-DL)
                tc, gaps, mask = Generate(F, winsize, LAMBDAS[k]-DL, T0=T0, TN=TN, dt=dt)
                STDS.append(np.std(tc[1][mask]))
                MAXS.append(np.max(tc[1][:]))
                TACTS_CDF = CDF(tc[1][mask])
                KSDS.append(max(abs(TACTS_CDF - NORM_CDF)))
                if report>1: plt.plot(LAMBDAS[-1],KSDS[-1],'or',ms=2)

    best = sorted(range(len(KSDS)), key=lambda k: KSDS[k])[0]
    if report:
        print("Optimal Lambda :",LAMBDAS[best], flush=True)
        print("Confidence Interval :",DL, flush=True)
        print('Search Size :',len(LAMBDAS), ' (BF : %d)'%int((Lmax-Lmin)/DL), flush=True)
    if report>1:
        plt.axvline(LAMBDAS[best], lw=2,alpha=0.3,color='g')
        plt.show()
        plt.figure(figsize=(12,3))
        output = Generate(F, winsize, LAMBDAS[best], T0=T0, TN=TN, dt=dt)[0]
        TACTS_CDF = CDF(output[1])
        plt.plot(TACTS_CDF,'-k', label='TACTS CDF')
        plt.plot(NORM_CDF,'--b', label='NORMAL CDF')
        plt.legend()
        plt.show()

    return LAMBDAS, KSDS, LAMBDAS[best], STDS[best], MAXS[best]

def OptimizeSpectrum(F, windowsizes, T0=None, TN=None, dt=None, njobs=4, Lmin=0.1, Lmax=20, ResZero=20, Focus=0.5, ResDepth=3, report = True, **kwargs):


    if report: print("Optimizing %d Windows"%len(windowsizes)," >> Res %d / Focus %.2f / Depth %d"%(ResZero,Focus,ResDepth),
            '| Workers :',njobs,end=' .. ',flush=True)

    NORM_CDF = normalCDF()

    OPT = Parallel(n_jobs=njobs) (
        delayed(OptimizeWindow)
        (F, windowsizes[i],T0=T0,TN=TN,dt=dt,Lmin=Lmin,Lmax=Lmax,
                ResZero=ResZero, Focus=Focus, ResDepth=ResDepth, NORM_CDF=NORM_CDF, report=1, **kwargs)
            for i in range(len(windowsizes)) )

    if report: print('Done.')
    return OPT

def displayOptimization(OPT, windowsizes, figsize=(12,10)):
    plt.figure(figsize=figsize)
    for Op,w in zip(OPT,windowsizes):
        plt.plot(Op[0],Op[1],'o', ms=1, label='%d: W=%.2f >> $\lambda$=%.2f'%(list(windowsizes).index(w),w,Op[2]), alpha=0.8)
        plt.axvline(Op[2], lw=1, alpha=0.2, color='g')
        plt.axhline(Op[1][Op[0].index(Op[2])],lw=1,alpha=0.2, color='r')
    plt.xlabel('$\lambda$')
    plt.ylabel('KSD')
    plt.legend(loc=1)
    plt.show()

def KS(X, report=True):
    NORM_CDF = normalCDF()
    TACTS_CDF = CDF(X)
    if report:
        plt.figure(figsize=(15,3))
        plt.plot(TACTS_CDF,'-k', label='TACTS CDF')
        plt.plot(NORM_CDF,'--b', label='NORMAL CDF')
        plt.legend()
        plt.show()
    return max(abs(TACTS_CDF - NORM_CDF))

# RECURRENCE METHODS

def RecurrenceWindows(F, frame, timeline, stdthr=0.2, report=1, framemask=None, tactsmask=None):
    average_sampling_time = np.diff(F[0,:]).mean()
    average_win_points = int(frame/average_sampling_time)
    P_diag = np.zeros(2*average_win_points)
    AdaptiveSTD = (stdthr * np.std(F[1,tactsmask]))
    if report:
        print("Recurrence Data :",F.shape[1]," /  frame : %d"%frame)
        print("Timeline :",len(timeline),'   Threshold :',AdaptiveSTD)
    DETS = []
    if framemask is None: framemask = np.ones(len(timeline), dtype=bool)
    for i in range(len(timeline)):
        T = timeline[i]
        if framemask[i]:
            if T-(frame/2)<F[0,0] or T+(frame/2)>F[0,-1]:
                DETS.append(-1)
            else :
                TSlice = np.where((F[0] >= T-(frame/2)) & (F[0] < T+(frame/2)))[0]
                RP = rp.RecurrencePlot(F[1,TSlice], threshold=AdaptiveSTD, silence_level=2)
                P_diag_temp = RP.diagline_dist()
                for j in range(len(P_diag_temp)):
                    P_diag[j] += P_diag_temp[j]
                DETS.append(RP.determinism(l_min=2))
        else:
            DETS.append(0)

    return DETS, P_diag

def SweepDeterminism(TACTSW, frame, timeline, report=1, njobs=4, timegaps=[],tactsgaps=[], **kwargs):
    if report: print('Compiling DETS (%d)'%len(TACTSW), end=' ')
    average_sampling_time = np.diff(TACTSW[0][0]).mean()
    average_win_points = int(frame/average_sampling_time)
    P_diags = np.zeros(shape=(len(TACTSW),2*average_win_points))
    DETS = np.zeros(shape=(len(TACTSW),len(timeline)))
    framemask = np.ones(len(timeline), np.bool)
    tactsmask = np.ones(len(TACTSW[0][0]), np.bool)
    for i in range(len(timeline)):
        T = timeline[i]
        for gap in timegaps:
            if T>=gap[0] and T<=gap[1]:
                framemask[i] = False
    for i in range(len(TACTSW[0][0])):
        for gap in tactsgaps:
            if i>=gap[0] and i<=gap[1]:
                tactsmask[i] = False

    for k in range(len(TACTSW)):
        if report: print('.',end='')
        DETlist, P_diags[k] = RecurrenceWindows(TACTSW[k], frame, timeline, report=0,
                    framemask=framemask, tactsmask=tactsmask, **kwargs)
        DETS[k] = np.array(DETlist)
    if report: print(' Done.')
    return DETS, P_diags

def CompileDETS(DETS):
    T = len(DETS[0])
    S = len(DETS)
    SDET = np.zeros(T)
    for t in range(T):
        for k in range(S):
            SDET[t] += DETS[k][t]
        SDET[t] /= S
    return SDET

def SignificanceDET(P, n, nofexp = 500, l_min=2, L=0.05, R=0.95, report=0, **kwargs):

    DET = np.zeros(nofexp)
    cdf = np.cumsum(P)
    cdf = cdf / float(cdf[-1])
    for i in range(nofexp):
        values = np.random.rand(n)
        value_bins = np.searchsorted(cdf, values)
        d = defaultdict(int)
        for j in value_bins: d[j]+=1
        y = max(d.keys())
        z = np.zeros(y+1)
        for k, v in d.items(): z[k] = v
        n_time = len(z)
        partial_sum = (np.arange(l_min,n_time) * z[l_min:]).sum()
        full_sum = (np.arange(n_time) * z).sum()
        DET[i] = partial_sum / float(full_sum)
        del d, z

    if report:
        plt.figure(figsize=(12,1))
        plt.hist(np.mean(DET)+np.diff(DET), bins=100)
        plt.show()
    return confidence(DET, L=L, R=R, **kwargs)

def confidence(x, L = 0.05, R = 0.95 ):

    hist, bins = np.histogram(x, bins=500)
    cdf = np.cumsum(hist)
    cdf = cdf / float(cdf[-1])
    ind = indices(cdf, lambda x: x >= L)
    left = bins[ind[0]]
    ind = indices(cdf, lambda x: x <= R)
    right = bins[ind[-1]]
    return left, right

def indices(a, func): return [i for (i, val) in enumerate(a) if func(val)]

def GetSpectrumConfidence(Pds, N, L=0.05, R=0.95, **kwargs):
    CONF = []
    print('Measuring Confidences (%d)'%len(Pds), end=' ')
    for k in range(len(Pds)):
        print('.', end='')
        Ndiag = int(sum(Pds[k])/N)
        CONF.append(SignificanceDET(Pds[k], Ndiag, L=L, R=R, **kwargs))
    print(' Done.')
    return CONF

def GetMeanConfidence(Conf):
    Mean=[0,0]
    for C in Conf:
        Mean[0] += C[0]/float(len(Conf))
        Mean[1] += C[1]/float(len(Conf))
    return Mean

# MISC METHODS

def addMeasurementNoise(F,stdnoise=0.1):
    N, FSTD = F.shape[1], np.std(F[1])
    Noise = np.random.uniform(low=-FSTD*stdnoise/2,high=FSTD*stdnoise/2,size=N)
    FN = np.empty(shape=(2,N))
    FN[0,:] = F[0,:]
    FN[1,:] = F[1,:]+Noise
    return FN

def RandomSampleSeries(F,n):
    D = F.shape[0]
    F_s = np.zeros(shape=(D,n))
    F_ss = np.zeros(shape=(D,n))
    t_s = random.sample([i for i in range(0,F.shape[1])],n)
    index = 0
    for t in t_s:
        F_s[:,index] = F[:,t]
        index += 1
    ind = np.argsort(F_s[0,:])
    index = 0
    for i in range(n):
        F_ss[:,index]=F_s[:,ind[index]]
        index += 1
    return F_ss

def CDF(x, bins=2000):
    CDFs = np.cumsum(np.histogram(x,bins)[0])
    return CDFs/CDFs[-1]

def normalCDF(size=500000, bins=2000):
    return CDF(np.random.normal(size=size),bins)

def displaySpectrum(timeline, DETS, figsize=(15,10), C=None):

    plt.figure(figsize=figsize)
    plt.axhline(0, color='k', alpha=0.2)
    for k in range(len(DETS)):
        plt.plot(timeline,k+DETS[k],'o', ms=1)
        if C is not None:
            plt.gca().axhspan(C[k][0]+k,C[k][1]+k, facecolor='k', alpha=0.3, edgecolor='k', linewidth = 0.0)
        plt.axhline(k+1, color='k',ls='--', alpha=0.4)
    plt.ylim(0,len(DETS))
    plt.show()

def plotSpikes(ax, Timeline, DET, C_down=None, C_up=None, colorL = '#ad2e24', colorH = '#14716c', colorM='0.1', stdthr=1, mask=None):

    if mask is None: mask = np.ones(len(Timeline), np.bool)
    if C_up is None: C_up = np.mean(DET[mask]) + stdthr * np.std(DET[mask])
    if C_down is None: C_down = np.mean(DET[mask]) - stdthr * np.std(DET[mask])
    ax.axhspan(C_down, C_up,facecolor=colorM, alpha=0.1,edgecolor='0.1', linewidth = 0.0)
    ax.fill_between(Timeline[mask], C_down, DET[mask], where=DET[mask]<=C_down, facecolor=colorL,
            edgecolor = colorL, interpolate=False,linewidth=1.0)
    ax.fill_between(Timeline[mask], C_up, DET[mask], where=DET[mask]>=C_up, facecolor=colorH,
            edgecolor = colorH, interpolate=False,linewidth=1.0)
