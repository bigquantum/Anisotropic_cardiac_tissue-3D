

set title 'Voltage in time Plot'
set grid
plot "datatime.dat" using 1:2 with lines
pause 1
reread
