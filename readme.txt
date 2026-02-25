
Difference Batch & SGD: Looking at all datapoints every time -> takes longer per descend but better overview
   vs looking at only one datapoint each descend -> less overview but better performance
The Gradient of course stays two-dimensional for this example. Basically what changes is the n in the sum that sums up the errors.

y = f(x) + e   where e is totally random
f(x) = bx     where it could also have a constant shift e.g. +2