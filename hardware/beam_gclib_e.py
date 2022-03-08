import sys
import string
import gclib

import casperfpga
import time
import numpy
import struct
import sys
import logging
import array
import os
import numpy as np
import datetime
import serial
import usb.core
import usb.util
import datetime
import getpass
import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab
import matplotlib.pyplot as plt
g = gclib.py() #make an instance of the gclib python class

g.GOpen('/dev/ttyUSB0 --direct --baud 19200')
print(g.GInfo())
#mid_b = 22.25 *12000#[in] to [counts]
#mid_b = .5 *12000#[in] to [counts]
c = g.GCommand #alias the command callable
def running_mean(x, N):
    return numpy.convolve(x, numpy.ones((N,))/N)[(N-1):]

def cut(cut,fre,dist):

    # Define constants:

    b_max = dist #[ft]
    b_min = - b_max
    a_max = 2*dist
    a_min = - a_max

    delta_a = 1 #[in]
    delta_b = 1 #[in]

    B_SP = 10000  #speed, 1000 [cts/sec]
    A_SP = 10000  #speed, 1000 [cts/sec]

    t_b = 2 #[sec]

    now = datetime.datetime.now()
    today = str(now.day) + '-' +str(now.month) + '-'+str(now.year)

    N_MULT = 9
    L_MEAN = 1
    N_TO_AVG = 1
    N_CHANNELS=21

    F_START= int(fre*1000./N_MULT) #in MHz
    F_STOP = F_START
    F_OFFSET = 10 #in MHz

    IGNORE_PEAKS_BELOW = int(738)
    IGNORE_PEAKS_ABOVE = int(740)
    ENDPOINT_DEC=2 #, always. according to syntonic user manual.
    ENDPOINT_HEX=0x02

    T_BETWEEN_DELTA_F = 0.5
    DELTA_T_USB_CMD = 0.5
    T_BETWEEN_SAMP_TO_AVG = 0.5
    T_TO_MOVE_STAGE = 3
    DELTA_T_VELMEX_CMD = 0.25

    X_MIN_ANGLE = b_min
    X_MAX_ANGLE =  b_max
    Y_MIN_ANGLE = a_min
    Y_MAX_ANGLE = a_max

    PHI_MIN_ANGLE =0
    PHI_MAX_ANGLE = 0 #90.
    DELTA_X_Y = delta_a

    if X_MIN_ANGLE == X_MAX_ANGLE:
        prodX = 1
    else:
        prodX = int(abs(X_MAX_ANGLE - X_MIN_ANGLE)/DELTA_X_Y +1)
    if Y_MIN_ANGLE == Y_MAX_ANGLE:
        prodY = 1
    else:
        prodY = int(abs(Y_MAX_ANGLE - Y_MIN_ANGLE)/DELTA_X_Y +1)
    if PHI_MIN_ANGLE == PHI_MAX_ANGLE:
        prodPHI = 1
    else:
        prodPHI = 1#int(abs(PHI_MAX_ANGLE - PHI_MIN_ANGLE)/DELTA_PHI + 1)

    nfreq = int(abs(F_START - F_STOP)*10 + 1)
    nsamp = int( prodX * prodY * prodPHI )
    print('nsamp = '+str(nsamp))
    STR_FILE_OUT = str(str(fre)+'GHz_2D_'+today+'.txt')
    arr2D_all_data=numpy.zeros((nsamp,(4*N_CHANNELS+5)))#, where the 5 extra are f,x,y,phi, index_signal of peak cross power in a single bin (where phase is to be measured)
    print(STR_FILE_OUT )
    REGISTER_LO_1 = 5000 #Labjack register number for the Labjack DAC0 output, which goes to LO_1.
    REGISTER_LO_2 = 5002 #Labjack register number for the Labjack DAC1 output, which goes to LO_2.

    ##################################################################
    #         Roach Definitions
    ##################################################################

    F_CLOCK_MHZ = 500
    f_max_MHz = (F_CLOCK_MHZ/4)
    KATCP_PORT=7147

    def set_RF_output(device,state): #for state, e.g. '1' for command '0x02' will turn ON the RF output.
        print('Setting RF output')
        n_bytes = 2 #number of bytes remaining in the packet
        n_command = 0x02 #the command number, such as '0x02' for RF output control.
        data=bytearray(64)
        data[0]=ENDPOINT_HEX # I do think this has to be included and here, because excluding endpoint as data[0] makes the synth not change its draw of current.
        data[1]=n_bytes
        data[2]=n_command
        data[3]=state
        LOs[int(device)].write(ENDPOINT_DEC,data)
        return

    def set_f(device,f): #sets frequency output of synth
        print('Setting frequency to '+str(f)+' MHz')
        n_bytes = 6 #number of bytes remaining in the packet
        n_command = 0x01 #the command number, such as '0x02' for RF output control.
        bytes = [hex(ord(b)) for b in struct.pack('>Q',(f*1.E6))]#Q is unsigned long long and has std size 8, we only ever use last 5 elements.
        data=bytearray(64)
        data[0]=ENDPOINT_HEX
        data[1]=n_bytes
        data[2]=n_command
        ISTRT=3
        print('in set_f, bytes = :'+str(bytes))
        ii = 0
        while (ii < 5):
                data[int(ii+ISTRT)]=int(bytes[ii+ISTRT],16)
                ii=ii+1

        LOs[int(device)].write(ENDPOINT_DEC,data)
        return

    def exit_fail():
        print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
        try:
            fpga.stop()
        except: pass
        raise
        exit()

    def exit_clean():
        try:
            fpga.stop()
        except: pass
        exit()

    def get_data(baseline):
        #print('   start get_data function')
        acc_n = fpga.read_uint('acc_num')
        print '   Grabbing integration number %i'%acc_n

        #get cross_correlation data...
        a_0r=struct.unpack('>512l',fpga.read('dir_x0_%s_real'%baseline,2048,0))
        a_1r=struct.unpack('>512l',fpga.read('dir_x1_%s_real'%baseline,2048,0))
        a_0i=struct.unpack('>512l',fpga.read('dir_x0_%s_imag'%baseline,2048,0))
        a_1i=struct.unpack('>512l',fpga.read('dir_x1_%s_imag'%baseline,2048,0))
        b_0i=struct.unpack('>512l',fpga.read('dir_x0_%s_imag'%baseline,2048,0))
        b_1i=struct.unpack('>512l',fpga.read('dir_x1_%s_imag'%baseline,2048,0))
        b_0r=struct.unpack('>512l',fpga.read('dir_x0_%s_real'%baseline,2048,0))
        b_1r=struct.unpack('>512l',fpga.read('dir_x1_%s_real'%baseline,2048,0))
        interleave_cross_a=[]
        interleave_cross_b=[]

        #get auto correlation data (JUST the A, B inputs)...
        a_0=struct.unpack('>512l',fpga.read('dir_x0_bb_real',2048,0))
        a_1=struct.unpack('>512l',fpga.read('dir_x1_bb_real',2048,0))
        b_0=struct.unpack('>512l',fpga.read('dir_x0_dd_real',2048,0))
        b_1=struct.unpack('>512l',fpga.read('dir_x1_dd_real',2048,0))
        interleave_auto_a=[]
        interleave_auto_b=[]

        #interleave cross-correlation and auto correlation data.
        for i in range(512):
            #cross
            interleave_cross_a.append(complex(a_0r[i], a_0i[i]))
            interleave_cross_a.append(complex(a_1r[i], a_1i[i]))
            interleave_cross_b.append(complex(b_0r[i], b_0i[i]))#For phase, new, test.
            interleave_cross_b.append(complex(b_1r[i], b_1i[i]))#For phase, new, test

            #auto
            interleave_auto_a.append(a_0[i])#'interleave' even and odd timestreams back into the original timestream (b.c. sampling rate is 2x your FPGA clock).
            interleave_auto_a.append(a_1[i])
            interleave_auto_b.append(b_0[i])#'interleave' even and odd timestreams back into the original timestream (b.c. sampling rate is 2x your FPGA clock).
            interleave_auto_b.append(b_1[i])

        #print('   end get_data function')
        return acc_n,interleave_cross_a,interleave_cross_b,interleave_auto_a,interleave_auto_b

    def drawDataCallback(baseline):
        #print('running get_data  function from drawDataCallback')
        acc_n,interleave_cross_a,interleave_cross_b,interleave_auto_a,interleave_auto_b= get_data(baseline)
        val=running_mean(numpy.abs(interleave_cross_a),L_MEAN)
        val[int(IGNORE_PEAKS_ABOVE):]=0
        val[: int(IGNORE_PEAKS_BELOW)]=0
        arr_index_signal = numpy.argpartition(val, -2)[-2:]
        index_signal = arr_index_signal[1]
        # IS THIS NECESSARY? Probably not here, at least. freq = numpy.linspace(0,f_max_MHz,len(numpy.abs(interleave_cross_a)))
        arr_ab = (numpy.abs(interleave_cross_a))
        arr_phase = (180./numpy.pi)*numpy.unwrap((numpy.angle(interleave_cross_b)))
        phase_signal = arr_phase[index_signal]
        arr_aa = (numpy.abs(interleave_auto_a))
        arr_bb = (numpy.abs(interleave_auto_b))

        #Only record relevant channels, right around peak:
        arr_aa = arr_aa[(index_signal - (N_CHANNELS/2)) : (1+index_signal + (N_CHANNELS/2))]
        arr_bb = arr_bb[(index_signal - (N_CHANNELS/2)) : (1+index_signal + (N_CHANNELS/2))]
        arr_ab = arr_ab[(index_signal - (N_CHANNELS/2)) : (1+index_signal + (N_CHANNELS/2))]
        arr_phase = arr_phase[(index_signal - (N_CHANNELS/2)) : (1+index_signal + (N_CHANNELS/2))]

        return running_mean(arr_aa,L_MEAN),running_mean(arr_bb,L_MEAN),running_mean(arr_ab,L_MEAN), arr_phase, index_signal

    def TakeAvgData():
        arr_phase= numpy.zeros((N_CHANNELS,1))
        arr_aa= numpy.zeros((N_CHANNELS,1))
        arr_bb= numpy.zeros((N_CHANNELS,1))
        arr_ab= numpy.zeros((N_CHANNELS,1))
        arr_index =numpy.zeros((1,1))

        arr2D_phase= numpy.zeros((N_TO_AVG,N_CHANNELS))#array of phase data, which I will take the mean of
        arr2D_aa=numpy.zeros((N_TO_AVG,N_CHANNELS))
        arr2D_bb=numpy.zeros((N_TO_AVG,N_CHANNELS))
        arr2D_ab=numpy.zeros((N_TO_AVG,N_CHANNELS))
        arr2D_index= numpy.zeros((N_TO_AVG,1))
        j = 0

        while (j < N_TO_AVG):
            arr2D_aa[j],arr2D_bb[j],arr2D_ab[j],arr2D_phase[j], arr2D_index[j]=drawDataCallback(baseline)
            #^^^^take in data from the roach. see function "drawDataCallback" above for how this works. "arr2D" array take in data across all frequency bins of the roach.
            j = j+1

        arr_phase=arr2D_phase.mean(axis=0)
        arr_aa=arr2D_aa.mean(axis=0)
        arr_bb=arr2D_bb.mean(axis=0)
        arr_ab=arr2D_ab.mean(axis=0)
        arr_index=arr2D_index.mean(axis=0)

        return arr_aa, arr_bb, arr_ab, arr_phase, arr_index

    def MakeBeamMap_e(i_f, f):
        i=0
        print('begin MakeBeamMap() for f = '+str(f))
        set_f(0,f)
        set_f(1,f + F_OFFSET)

        arr_phase= numpy.zeros((N_CHANNELS,1))
        arr_aa= numpy.zeros((N_CHANNELS,1))
        arr_bb= numpy.zeros((N_CHANNELS,1))
        arr_ab= numpy.zeros((N_CHANNELS,1))
        index_signal = 0

        B_SP = 8000
        #Motion Complete
        print('Going to starting position...')

        c = g.GCommand #alias the command callable

        c('AB') #abort motion and program
        c('MO') #turn off all motors
        c('SHB') #servo B
        c('SPB='+str(B_SP))
        c('PRB='+str(-b_max*12000)) #relative move
        #print(' Starting move...')
        c('BGB') #begin motion
        g.GMotionComplete('B')

        c('SHA') #servo B
        c('SPA='+str(A_SP))
        c('PRA='+str(a_min*7000)) #relative move
        #print(' Starting move...')
        c('BGA') #begin motion
        g.GMotionComplete('A')
        i=0
        phi = 0
        for a in numpy.arange (a_min,a_max+delta_a,delta_a):

            for b in numpy.arange (-b_max,b_max+delta_b,delta_b):

                time.sleep(DELTA_T_VELMEX_CMD)

                phi = 90

                print(' Recording data: f: ('+str(f)+'/'+str(F_STOP)+'), x: ('+str(a)+'/'+str(X_MAX_ANGLE)+'), y: ('+str(b)+'/'+str(Y_MAX_ANGLE))
                arr_aa, arr_bb, arr_ab, arr_phase,index_signal = TakeAvgData()
                arr2D_all_data[i] = ([f]+[a]+[b]+[phi]+[index_signal]+arr_aa.tolist()+arr_bb.tolist()+arr_ab.tolist()+arr_phase.tolist())
                i = i+1
                print('    ...done. ('+str(i)+'/'+str(nsamp)+')')
                time.sleep(DELTA_T_VELMEX_CMD)

                if b<b_max:
                    c('SHB') #servo B
                    c('SPB='+str(B_SP))
                    c('PRB='+str(delta_b*12000)) #relative move
                    #print(' Starting move...')
                    c('BGB') #begin motion
                    g.GMotionComplete('B')
                    #print(str(b+1)+'B done.')

            time.sleep(DELTA_T_VELMEX_CMD)
            c('SPB='+str(B_SP))
            c('PRB='+str(-2*b_max*12000)) #relative move, 3000 cts
            print(' RETURNING B HOME...')
            c('BGB') #begin motion
            g.GMotionComplete('B')
            print('B @ HOME.')
            if a<a_max:
                c('SHA') #servo A
                c('SPA='+str(A_SP)) #speead, 1000 cts/sec
                c('PRA='+str(delta_a*7000)) #relative move, 3000 cts
                #print(' Starting move...')
                c('BGA') #begin motion
                g.GMotionComplete('A')
                #print(str(a+1)+'A done.')
                #del c #delete the alias

        # move back to center position

        c('AB') #abort motion and program
        c('MO') #turn off all motors
        c('SHB') #servo B
        c('SPB='+str(22000))

    #     c('HMB')
    #     c('BGB') #begin motion
    #     g.GMotionComplete('B')
        c('PRB='+str(b_max*12000)) #relative move
        #print(' Starting move...')
        c('BGB') #begin motion
        g.GMotionComplete('B')

    #     mid_a = 19.7 *7000#[in] to [counts]

        c('SHA') #servo B
        c('SPA='+str(22000))

    #     c('HMA')
    #     c('BGA') #begin motion
    #     g.GMotionComplete('A')
        c('PRA='+str(-a_max*7000)) #relative move
        #print(' Starting move...')
        c('BGA') #begin motion
        g.GMotionComplete('A')

    #     print('A @ HOME.')
        del c #delete the alias
        time.sleep(DELTA_T_VELMEX_CMD)

        print(' end f = '+str(f))

    # debug log handler
    class DebugLogHandler(logging.Handler):
        """A logger for KATCP tests."""

        def __init__(self,max_len=100):
            """Create a TestLogHandler.
                @param max_len Integer: The maximum number of log entries
                                        to store. After this, will wrap.
            """
            logging.Handler.__init__(self)
            self._max_len = max_len
            self._records = []

        def emit(self, record):
            """Handle the arrival of a log message."""
            if len(self._records) >= self._max_len: self._records.pop(0)
            self._records.append(record)

        def clear(self):
            """Clear the list of remembered logs."""
            self._records = []

        def setMaxLen(self,max_len):
            self._max_len=max_len

        def printMessages(self):
            for i in self._records:
                if i.exc_info:
                    print '%s: %s Exception: '%(i.name,i.msg),i.exc_info[0:-1]
                else:
                    if i.levelno < logging.WARNING:
                        print '%s: %s'%(i.name,i.msg)
                    elif (i.levelno >= logging.WARNING) and (i.levelno < logging.ERROR):
                        print '%s: %s'%(i.name,i.msg)
                    elif i.levelno >= logging.ERROR:
                        print '%s: %s'%(i.name,i.msg)
                    else:
                        print '%s: %s'%(i.name,i.msg)
    #START OF MAIN:

    if __name__ == '__main__':

        from optparse import OptionParser


        p = OptionParser()
        p.set_usage('poco_init_no_quant2.py')
        p.set_description(__doc__)
        p.add_option('-s', '--skip', dest='skip', action='store_true',
            help='Skip reprogramming the FPGA and configuring EQ.')
        p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=0.5*(2**28)/2048, #for low pass filter and amplifier this seems like a good value, though not tested with sig. gen. #        25 jan 2018: 0.01
            help='Set the number of vectors to accumulate between dumps. default is 0.5*(2^28)/2048.')#for roach full setup.

        p.add_option('-c', '--cross', dest='cross', type='str',default='bd',
            help='Plot this cross correlation magnitude and phase. default: bd')
        p.add_option('-g', '--gain', dest='gain', type='int',default=2,
            help='Set the digital gain (4bit quantisation scalar). default is 2.')
        p.add_option('-f', '--fpg', dest='fpgfile', type='str', default='',
            help='Specify the bof file to load')


        opts, args = p.parse_args(sys.argv[1:])
        roach='192.168.4.20'

        baseline=opts.cross

    try:
        loggers = []
        lh=DebugLogHandler()
        logger = logging.getLogger(roach)
        logger.addHandler(lh)
        logger.setLevel(10)

        print('Connecting to server %s ... '%(roach)),

        #fpga = casperfpga.CasperFpga(roach)
        fpga = casperfpga.katcp_fpga.KatcpFpga(roach)
        time.sleep(3)

        if fpga.is_connected():
            print 'ok\n'
        else:
            print 'ERROR connecting to server %s.\n'%(roach)
            exit_fail()

        ### #prepare synths ###
        LOs = tuple(usb.core.find(find_all=True, idVendor=0x10c4, idProduct=0x8468))
        print LOs[0].bus, LOs[0].address
        print LOs[1].bus, LOs[1].address

        if ((LOs[0] is None) or (LOs[1] is None)): #Was device found?
            raise ValueError('Device not found.')
        else:
            print(str(numpy.size(LOs))+' device(s) found:')
        ii=0
        while (ii< np.size(LOs)):
                LOs[ii].reset()
                reattach = False #Make sure the USB device is ready to receive commands.
                if LOs[ii].is_kernel_driver_active(0):
                        reattach = True
                        LOs[ii].detach_kernel_driver(0)
                LOs[ii].set_configuration()
                ii=ii+1

        set_RF_output(0,1) #Turn on the RF output. (device,state)
        set_RF_output(1,1)
        ### end synth prep ###

        i = 0

        f_sample = F_START#(((VfreqSet-1.664)/dv_over_df)*(12.0) + 120.0)
        print('Begining step '+str(i)+' of '+str(nfreq)+', where frequency = '+str(f_sample))
        time.sleep(T_BETWEEN_DELTA_F)
        MakeBeamMap_e(0, f_sample)
        print('yes')

        ##
        print('Beam Map Complete.')

        arr2D_all_data = numpy.around(arr2D_all_data,decimals=3)
        print('Saving data...')
        numpy.savetxt(STR_FILE_OUT,arr2D_all_data,fmt='%.3e',header=('f_sample(GHz), x, y, phi, index_signal of peak cross power, and '+str(N_CHANNELS)+' points of all of following: aa, bb, ab, phase (deg.)'))

        print('Done. Exiting.')

    except KeyboardInterrupt:
        exit_clean()

    return STR_FILE_OUT
beam = cut('e',95,5)
