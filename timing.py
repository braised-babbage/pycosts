# timing.py
#
# Routines to help in timing the execution of
# various code fragments or routines, and to
# infer a good formula for the resulting runtimes.
#
# Original code by Ronald L. Rivest, 2007
#
# Cleaned up and ported to Python 3 by Erik Davis, 2016


import math
import scipy.linalg
import string
import sys
import timeit
from collections import namedtuple

# Parameter generation routines

def lg(x):
    return math.log(x)/math.log(2.0)

def sqrt(x):
    return math.sqrt(x)

def make_param_list(spec_string,growth_factor):
    """
    Generate a list of dictionaries
    given maximum and minimum values for each range.
    Each min and max value is a *string* that can be evaluted;
    each string may depend on earlier variable values 
    Values increment by factor of growth_factor from min to max
    Example:
       make_param_list("1<=n<=1000")
       make_param_list("1<=n<=1000;1<=m<=1000;min(n,m)<=k<=max(n,m)")
    """
    var_list = []       
    spec_list = spec_string.split(";")
    D = {}
    D['lg']=lg
    D['sqrt'] = sqrt
    D_list = [D]
    for spec in spec_list:
        spec_parts = spec.split("<=")
        assert(len(spec_parts)==3)
        lower_spec = spec_parts[0]
        var_name = spec_parts[1]
        assert(len(var_name)==1)
        var_list.append(var_name)
        upper_spec = spec_parts[2]
        new_D_list = []
        for D in D_list:
            new_D = D.copy()
            val = eval(lower_spec,D)
            while val<=eval(upper_spec,D):
                new_D[var_name] = val
                new_D_list.append(new_D.copy())
                val *= growth_factor
        D_list = new_D_list
    # for D in D_list: print D
    return (var_list,D_list)

# sample("1<=n<=1000;1<=m<=1000;min(n,m)<=k<=max(n,m)",2)

def fit(var_list,param_list,run_times,f_list):
    """
    Return matrix A needed for least-squares fit.
    Given:
        list of variable names
        list of sample dicts for various parameter sets
        list of corresponding run times
        list of functions to be considered for fit
            these are *strings*, e.g. "n","n**2","min(n,m)",etc.
    prints:
        coefficients for each function in f_list
    """
    print("var_list",var_list)
    print("Function list:",f_list)
    print("run times:")
    for i in range(len(param_list)):
        for v in var_list:
            print(v,"= %6s"%param_list[i][v], end="")
        print(": %8f"%run_times[i],"microseconds")
        # print "  n = %(n)6s"%param_list[i],run_times[i],"microseconds"
    rows = len(run_times)
    cols = len(f_list)
    A = [ [0 for j in range(cols)] for i in range(rows) ]
    for i in range(rows):
        D = param_list[i]
        for j in range(cols):
            A[i][j] = float(eval(f_list[j],D))
    b = run_times
    # print "A:"
    # print A
    # print "b:"
    # print b

    # (x,resids,rank,s) = scipy.linalg.lstsq(A,b)
    (x,resids,rank,s) = fit2(A,b)

    print("Coefficients as interpolated from data:")
    for j in range(cols):
        sign = ''
        if x[j]>0 and j>0: 
            sign="+"
        elif x[j]>0:
            sign = " "
        print("%s%g*%s"%(sign,x[j],f_list[j]))

    print("(measuring time in microseconds)")
    print("Sum of squares of residuals:",resids)
    print("RMS error = %0.2g percent"%(math.sqrt(resids/len(A))*100.0))
    # print "Rank:",rank
    # print "SVD:",s
    sys.stdout.flush()
    
import scipy.optimize

def fit2(A,b):
    """ Relative error minimizer """
    def f(x):
        assert(len(x) == len(A[0]))
        resids = []
        for i in range(len(A)):
            sum = 0.0
            for j in range(len(A[0])):
                sum += A[i][j]*x[j]
            relative_error = (sum-b[i])/b[i]
            resids.append(relative_error)
        return resids
    ans = scipy.optimize.leastsq(f,[0.0]*len(A[0]))
    # print "ans:",ans
    if len(A[0])==1:
        x = [ans[0]]
    else:
        x = ans[0]
    resids = sum([r*r for r in f(x)])
    return (x,resids,0,0)

Test = namedtuple('Test', ['name', 'doc', 'spec', 'trials',
                           'growth_factor', 'expr', 'setup',
                           'f_list'])

def run_test(test):
    print("\nTest %s: %s" % (test.name, test.doc))
    
    print("Spec_string: ",test.spec,"by factors of",test.growth_factor)
    var_list,param_list = make_param_list(test.spec,
                                          test.growth_factor)
    run_times = []
    for D in param_list:
        t = timeit.Timer(test.expr.format_map(D), test.setup.format_map(D))
        run_times.append(t.timeit(test.trials)*1e6/float(test.trials))

    fit(var_list,param_list,run_times,test.f_list)



misc_tests = [Test("Misc-2",
                   "pass",
                   spec="10000<=n<=1000000",
                   trials=1000,
                   growth_factor=2,
                   expr="pass",
                   setup="",
                   f_list=("1",))]

number_tests = [Test("Number-1",
                     "time to compute int('1'*n)",
                     spec="1000<=n<=10000",
                     growth_factor=2,
                     f_list = ("n**2",),
                     trials=1000,
                     expr="int(x)",
                     setup="import string;x='1'*{n}"),
                Test("Number-2",
                     "time to compute repr(2**n)",
                     spec="1000<=n<=10000",
                     growth_factor=2,
                     f_list=("n**2",),
                     trials=1000,
                     expr="repr(x)",
                     setup="x=2**{n}"),
                Test("Number-3",
                     "time to convert (2**n) to hex",
                     spec="1000<=n<=100000",
                     growth_factor=2,                 
                     f_list = ("n",),
                     trials=1000,
                     expr="hex(x)",
                     setup="x=2**{n}"),
                Test("Number-4",
                     "time to add 2**n to itself",
                     spec="1000<=n<=1000000",
                     growth_factor = 2,
                     f_list = ("n",),
                     trials=10000,
                     expr="x+x",
                     setup="x=2**{n}"),
                Test("Number-5",
                     "time to multiply (2**n//3) by itself",
                     spec="1000<=n<=100000",
                     growth_factor = 2,
                     f_list = ("n**1.585",),
                     trials=1000,
                     expr="x*x",
                     setup="x=(2**{n})//3"),
                Test("Number-6",
                     "time to divide (2**(2n) by (2**n))",
                     spec="1000<=n<=50000",
                     growth_factor=2,
                     f_list = ("n**2",),
                     trials=1000,
                     expr="w//x",
                     setup="w=(2**(2*{n}));x=(2**({n}))"),
                Test("Number-7",
                     "time to compute remainder of (2**(2n) by (2**n))",
                     spec="1000<=n<=50000",
                     growth_factor = 2,
                     f_list=("n**2",),
                     trials=1000,
                     expr="w % x",
                     setup="w=(2**(2*{n}));x=(2**({n}))"),
                Test("Number-8",
                     "time to compute pow(x,y,z)",
                     spec="1000<=n<=5000",
                     growth_factor=2,
                     f_list=("n**3",),
                     trials=10,
                     expr="pow(x,y,z)",
                     setup="z=(2**{n})+3;x=y=(2**{n})+1"),
                Test("Number-9",
                     "time to compute 2**n",
                     spec="1000<=n<=1000000",
                     growth_factor=2,
                     f_list=("1",),
                     trials=10000,
                     expr="2**{n}",
                     setup="")]

string_tests = [Test("String-1",
                     "extract a byte from a string",
                     spec="1000<=n<=1000000",
                     growth_factor=2,
                     f_list=("1",),
                     trials=1000,
                     expr="s[500]",
                     setup="s='0'*{n}"),
                Test("String-2",
                     "concatenate two string of length n",
                     spec="1000<=n<=500000",
                     growth_factor=2,
                     f_list=("n",),
                     trials=1000,                      
                     expr="s+t",
                     setup ="s=t='0'*{n}"),
                Test("String-3",
                     "extract a string of length n/2",
                     spec="1000<=n<=500000",
                     growth_factor=2,
                     f_list=("n",),
                     trials=1000,
                     expr="s[0:{n}//2]",
                     setup="s='0'*{n}"),
                Test("String-4",
                     "translate a string of length n",
                     spec= "1000<=n<=500000",
                     growth_factor = 2,
                     f_list = ("n",),
                     trials = 1000,
                     expr="s.translate(T)",
                     setup=("s='0'*{n};import string;"
                            "T=dict(); T[ord('1')] = '2'"))]

list_tests = [Test("List-1",
                   "create an empty list",
                   spec="1<=n<=10",
                   growth_factor = 2,
                   f_list = ("1",),
                   trials = 1000,
                   expr="x = list()",
                   setup=""),
              Test("List-2",
                   "list (array) lookup",
                   spec="10000<=n<=1000000",
                   growth_factor = 2,
                   f_list = ("1",),
                   trials = 1000,
                   expr="x=L[5]",
                   setup="L=[0]*{n}"),
              Test("List-3",
                   "appending to a list of length n",
                   spec="10000<=n<=1000000",
                   growth_factor = 2,
                   f_list = ("1"),                
                   trials = 1000,
                   expr="L.append(0)",
                   setup="L=[0]*{n};L.append(0)"),
              Test("List-4",
                   "Pop",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("1",),
                   trials = 200,
                   expr="L.pop()",
                   setup="L=[0]*{n}"),
              Test("List-5",
                   "concatenating two lists of length n",
                   spec= "1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n",),
                   trials = 2000,
                   expr="L+L",
                   setup="L=[0]*{n}"),
              Test("List-6",
                   "extracting a slice of length n/2",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n",),
                   trials = 2000,
                   expr="L[0:{n}//2]",
                   setup="L=[0]*{n}"),
              Test("List-7",
                   "copy",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n",),
                   trials = 2000,
                   expr="L[:]",
                   setup="L=[0]*{n}"),
              Test("List-8",
                   "Assigning a slice of length n/2",
                   spec= "1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n",),
                   trials = 2000,
                   expr="L[0:{n}//2]=L[1:1+{n}//2]",
                   setup="L=[0]*{n}"),
              Test("List-9",
                   "Delete first",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n",),
                   trials = 200,
                   expr="del L[0]",
                   setup="L=[0]*{n}"),
              Test("List-10",
                   "Reverse",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n",),
                   trials = 200,
                   expr="L.reverse()",
                   setup="L=[0]*{n}"),
              Test("List-11",
                   "Sort",
                   spec = "1000<=n<=100000",                 
                   growth_factor = 2,
                   f_list = ("n*lg(n)",),
                   trials = 200,
                   expr="L.sort()",
                   setup=("import random;"
                          "L=[random.random() for i in range({n})]"))]

dict_tests = [Test("Dict-1",
                   "create an empty dictionary",                      
                   spec="1<=n<=1",
                   growth_factor = 2,
                   f_list = ("1",),
                   trials = 1000,
                   expr = "x = dict()",
                   setup=""),
              Test("Dict-2",
                   "dictionary lookup",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("1",),
                   trials = 1000,
                   expr="x = d[1]",
                   setup="d = dict([(i,i) for i in range({n})])"),
              Test("Dict-3",
                   "dictionary copy",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n",),
                   trials = 1000,
                   expr="d.copy()",
                   setup="d = dict([(i,i) for i in range({n})])"),
              Test("Dict-4",
                   "dictionary list items",
                   spec="1000<=n<=100000",
                   growth_factor = 2,
                   f_list = ("n*lg(n)",),
                   trials = 1000,
                   expr="d.items()",
                   setup="d = dict([(i,i) for i in range({n})])")]

if __name__ == "__main__":
    for test in number_tests + string_tests + list_tests + dict_tests:
        run_test(test)

