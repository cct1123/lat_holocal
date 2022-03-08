import gclib

g = gclib.py() #make an instance of the gclib python class

g.GOpen('/dev/ttyUSB0 --direct --baud 19200')
print(g.GInfo())
#mid_b = 22.25 *12000#[in] to [counts]
#mid_b = .5 *12000#[in] to [counts]
c = g.GCommand #alias the command callable


B_SP = 10000  #speed, 1000 [cts/sec]
A_SP = 10000  #speed, 1000 [cts/sec]


print('Going to starting position...')

c('AB') #abort motion and program
c('MO') #turn on all motors
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
