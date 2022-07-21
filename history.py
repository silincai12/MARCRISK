                   GroundFails = GroundFails +  simgroundFails
               if GroundFails >= 3 or FlightFails >= 1 :
                   fvec[z] = 1
           fmat[b:,] = fvec
       #testvec = np.delete(fmat,0,0)
       #bigp = testvec.mean(axis = 0)
       bigp = fmat.mean(axis = 0)
       return(bigp)


test11 = detect(ages,numComp, numOldFlightTests, numMidFlightTests, numYoungFlightTests, numOldGroundTests,numMidGroundTests, numYoungGroundTests, outyears, failureModel, numnew, flightEff, groundEff,numsims, sampEff, seed, agestarts)
test12 = detect(ages2,numComp2, numOldFlightTests, numMidFlightTests, numYoungFlightTests, numOldGroundTests,numMidGroundTests, numYoungGroundTests, outyears, failureModel, numnew2, flightEff2, groundEff2,numsims, sampEff, seed, agestarts2)

#Risk.model 





import pandas as pd
import numpy as np
import math
import random

def riskmodel(ages, numComp, numOldFlightTests, numMidFlightTests, numYoungFlightTests, numOldGroundTests, numMidGroundTests, numYoungGroundTests, failureModel, outyears, flightEff, groundEff, sampEff, threshold, yearsdelay, numnew, Retire, numrepairs, startca,
               numsims, seed, agestarts,
               plot=True, plotHist=True, plotRisk=True, plotDetect=True, plotUnreliability=True,
               unreliabilityThreshold=1, replaceOldPlots=True,
               pdfOfPlots=False):
 numFlightTest = sum([numOldFlightTests, numMidFlightTests, numYoungFlightTests])
 numGroundTest = sum([numOldGroundTests, numMidGroundTests, numYoungGroundTests])

 if numsims < 1:
  return 'numsims must be a positive whole number'
 if numnew < 0 or numnew != round(numnew):
  return 'numnew must be a nonnegative number'


 leftover = sum(np.array(numComp) - (numFlightTest - numnew)*(outyears -1))

 insert1 = np.repeat(np.array([1,0]), [leftover%3, 3-leftover%3])

 if sum((leftover//3 + insert1) < [numOldFlightTests, numMidFlightTests, numYoungFlightTests])>0 or sum((leftover - numFlightTest)//3 + insert1< [numOldGroundTests, numMidGroundTests, numYoungGroundTests])>0 :
     return print('Error: By performing',numFlightTest, "flight tests per year - choosing", numOldFlightTests, "old", numMidFlightTests, "mid, and", numYoungFlightTests, "young - and running",  numGroundTest, "ground tests:", numOldGroundTests, "old", numMidGroundTests, "mid, and", numYoungGroundTests, "young (and adding", numnew, "new components each year)", "you have only enough samples for about", math.floor(1+(sum(numComp) - max(max(3* np.array([numOldFlightTests, numMidFlightTests, numYoungFlightTests])- np.array([2,1,0])), max(3* np.array([numOldFlightTests, numMidFlightTests, numYoungFlightTests])- np.array([2,1,0]))+ numFlightTest))/(numFlightTest-numnew)), "years into the future")

 if sum([numOldGroundTests, numMidGroundTests, numYoungGroundTests]) > sum(numComp):
     return print("You cannot perform", numGroundTest, "groundtests per year because only have", sum(numComp), "components in your inventory")

 df = pd.DataFrame({
 'ages': ages,
 'numComp': numComp 
        })
 df

 df = df.sort_values(by=['ages'])

 agevec = df['ages']
 countvec = df['numComp'] 


 prob= detect(ages,numComp, numOldFlightTests, numMidFlightTests, numYoungFlightTests, numOldGroundTests,numMidGroundTests, numYoungGroundTests, outyears, failureModel, numnew, flightEff, groundEff,numsims, sampEff, seed, agestarts)

 if outyears != round(outyears):
  return 'outyears should be a whole number representing the number of years into the future to simulate'
 if max(prob) < threshold:
  print('Probability of at least one detected flight failure OR at least 3 detected ground failures does not reach threshold so no corrective action takes place')
 YearsDetect =min(np.array(( np.where(prob == min(np.extract(prob>threshold, prob))))))
 YearsDetect = YearsDetect[0]
 StartCorrect = YearsDetect + yearsdelay

#If you want to manually set what year to start corrective action
 if startca != None:
  if startca < 1 or startca > outyears or startca != int(startca):
   return(print("You set startca =", startca, "but it needs to be a whole number between 1 and the number of outyears. To let the model decide when corrective action takes place set startca = None" ))
  else:
   StartCorrect = startca

 if StartCorrect > outyears:
  print('No corrective action take place because the years to corrective action', StartCorrect, "exceeds the number of years being simulated" )

 if np.any((np.array(numrepairs) == np.rint(np.array(numrepairs)) )):
  'numrepairs is not a vetor of non-negative whole numbers'

 if np.any(np.array(numrepairs) < 0):
  return('Some entries for numrepairs is negative. The number of repairs should be positive, whole numbers.')

 if yearsdelay <0 or yearsdelay != round(yearsdelay):
  return('yearsdelay must be a non-negative whole number')

 if threshold <0 or threshold >1:
  return('threshold must be a number between 0 and 1 (inclusive)')

 if Retire < 0 or Retire  != round(Retire):
  return("numtoretire must be a non-negative whole numbers")

 #unrelMat = np.empty(shape = (len(numrepairs), outyears), dtype = 'object')
 rIndex = -1
#gvec = np.zeros((min(StartCorrect-1, outyears)))
 gvec = np.zeros(outyears)
 unrel = np.empty(shape = (1,outyears,len(numrepairs)))
 for r in range(0, len(numrepairs)): 
  a = numrepairs[r] 
  rIndex = rIndex + 1
  Gmat = np.empty(shape = (numsims, outyears))
  for b in range(0, numsims):
   inventory = np.array(np.repeat(agevec, countvec)) 
   for z in range(0, min(StartCorrect, outyears)):
    inventory = inventory +1 
    inventory = np.concatenate((np.repeat(0, numnew), np.array(inventory)))
    FindexYoung = random.sample(range(1, math.ceil(len(inventory)/3)),numYoungFlightTests)
    FindexMid = random.sample(range(math.ceil(len(inventory)/3 +1 ),math.ceil(2*len(inventory)/3)),numMidFlightTests)
    FindexOld = random.sample(range(math.ceil(2*len(inventory)/3 +1 ),len(inventory)),numOldFlightTests)



    DelIndex = FindexYoung + FindexMid + FindexOld
    inventory = np.delete(inventory, DelIndex)
    from functools import partial
    partial_func = partial(failureModel, agestarts = agestarts,rate = rate)
    estFailProb = np.array(list(map(partial_func, inventory)))
    gvec[z] = np.mean(estFailProb)
   if StartCorrect < outyears:
    for z in range(StartCorrect, outyears):
     inventory = inventory +1
     inventory = np.concatenate(( np.repeat(0, numnew), np.array(inventory)))

     if Retire >0:
      RetireInd = random.sample(range(math.ceil(len(inventory)*(4/5) +1 ), len(inventory)), min(sum(inventory !=0), Retire))
      inventory = np.delete(inventory, RetireInd)
     if a > 0:
      RepairInd = random.sample(range(math.ceil(len(inventory)*(4/5) +1 ), len(inventory)), min(sum(inventory !=0), a))
      inventory = np.concatenate((np.repeat(0, a), np.delete(inventory, RepairInd)))

     FindexYoung = random.sample(range(1, math.ceil(len(inventory)/3)),numYoungFlightTests)
     FindexMid = random.sample(range(math.ceil(len(inventory)/3 +1 ),math.ceil(2*len(inventory)/3)),numMidFlightTests)
     FindexOld = random.sample(range(math.ceil(2*len(inventory)/3 +1 ),len(inventory)),numOldFlightTests)


     DelIndex = FindexYoung + FindexMid + FindexOld
     inventory = np.delete(inventory, DelIndex)
     estFailProb = np.array(list(map(partial_func, inventory)))
     gvec[z] = np.mean(estFailProb)
    Gmat[b:,] = gvec
  unrel[...,rIndex] = Gmat.mean(axis = 0)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 from datetime import date
 def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)
 if plot == True:
  today = date.today()
  yr = today.year #yr: current year
  timeDet = min(which(prob > threshold))
  projYr = [i + yr for i in list(np.arange(1,outyears+1,1))]

 if plotHist == True: #Histogram of Initial Component Ages
  ageVec = ages
  countVec = numComp
  total = sum(countVec)

  Hplot = plt.figure(figsize=(10, 5))
  plt.hist(np.repeat(ageVec, countVec), color = "darkgrey", edgecolor = "black",density=True)
  plt.title("Histogram of Initial Component Ages")
  plt.xlabel("Initial Ages")
  plt.ylabel("Density")
  plt.legend(["Total # of Components = " + str(total)], loc = "upper left")
  plt.show()


 if plotRisk == True:




  Rplot = plt.figure(figsize=(15, 8))
  compAge = np.arange(0, max(ages)+outyears)
  from functools import partial
  failureModel = piecewise
  partial_func = partial(failureModel, agestarts = agestarts, rate = rate)
  estFailProb = np.array(list(map(partial_func, compAge)))
  plt.plot(compAge,estFailProb)
  plt.xlabel("Component Age")
  plt.ylabel("Probability of Failure")
  plt.title("Failure Probability Model")
  plt.legend(["age starts = " + str(agestarts)+ "\n rate = "+ str(rate)], loc = "upper left")
  plt.show()


 if plotDetect == True: 
  projYr = [i + yr for i in list(np.arange(1,outyears+1,1))]
  plt.plot(projYr,prob)
  plt.xlabel('Year')
  plt.ylabel("Cumulative Probability")
  plt.title("Probabillity of Detection")
  plt.show()
 if sum(prob >= threshold) > 0:
  Dplot, ax = plt.subplots(figsize=(8, 4), layout='constrained')
  ax.plot(projYr,prob, linestyle = "solid")
  plt.hlines(y = threshold, xmin = yr, xmax = yr + timeDet, linestyle = "dashed", color = 'black')
  plt.vlines(x = yr + timeDet, ymin = 0 , ymax = threshold , linestyle = "dashed", color = 'black')
  plt.text(2025,threshold + 0.05,threshold)
  plt.text(yr + timeDet + 1, 0.5, "Time of Detection")
  plt.xlabel("Year")
  plt.ylabel("Cumulative Probability")
  plt.title("Probability of Detection")
  plt.show()


 if plotUnreliability == True:

  fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
  for i in range(0,len(numrepairs)):
   y = "Scenario="+str(i+1)
   ax.plot(projYr, np.transpose(unrel[...,i]), label = ("Repair "+ str(numrepairs[i])+ " per year"))
  ax.set_xlabel('Year')  # Add an x-label to the axes.
  ax.set_ylabel('Average Failure Probability')  # Add a y-label to the axes.
  ax.set_title("Unreliability Over Time with Correctie ACtion")  # Add a title to the axes.



 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in numrepairs: 
  keys[y] = "Unreliability, repairs="+str(i)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = unrel[...,i]
 output["YearsToDetect"] = YearsDetect
 output['StartCorrect'] = StartCorrect
 output["Detect Probability"] = prob
 output['outyears'] = outyears
 output['number of repairs'] = numrepairs
 output['Histogram'] = Hplot
 output['Failure Model'] = Rplot
 output['Detection Plot'] = Dplot
 output['Unreliability Plot'] = fig
 return(output)



test = riskmodel(ages,numComp, numOldFlightTests, numMidFlightTests, numYoungFlightTests, numOldGroundTests,numMidGroundTests, numYoungGroundTests, failureModel, outyears,  flightEff, groundEff, sampEff, threshold, yearsdelay, numnew, Retire, numrepairs, startca, numsims, seed,agestarts)
test2 = riskmodel(ages2,numComp2, numOldFlightTests, numMidFlightTests, numYoungFlightTests, numOldGroundTests,numMidGroundTests, numYoungGroundTests, failureModel, outyears,  flightEff2, groundEff2, sampEff, threshold, yearsdelay, numnew2, Retire2, numrepairs2, startca, numsims, seed, agestarts2) 
test3 = riskmodel(ages3,numComp3, numOldFlightTests, numMidFlightTests, numYoungFlightTests, numOldGroundTests,numMidGroundTests, numYoungGroundTests, failureModel, outyears,  flightEff3, groundEff3, sampEff, threshold, yearsdelay, numnew3, Retire3, numrepairs3, startca, numsims, seed, agestarts3) 
comps = [test,test2,test3]
compnames = ["test", "test2", "test3"]
#Combined Unreliability

def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))


 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
f['Combined Unreliability']
f
keys
test['StartCorrect]
test['StartCorrect']
def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))

 StartCA = 5
 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  StartCA[i] = compnames, comps['StartCorrect'][i] + year
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig
 output['Years to start CA'] = StartCA

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))

 StartCA = 5
 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  StartCA[i] = compnames+ comps['StartCorrect'][i] + year
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig
 output['Years to start CA'] = StartCA

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
keys
def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))

 StartCA = None * len(comps)
 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  StartCA[i] = compnames+ comps['StartCorrect'][i] + year
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig
 output['Years to start CA'] = StartCA

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))

 StartCA = [None] * len(comps)
 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  StartCA[i] = compnames+ comps['StartCorrect'][i] + year
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig
 output['Years to start CA'] = StartCA

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
compnames
def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))

 StartCA = [None] * len(comps)
 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  StartCA[i] = compnames[i] + comps['StartCorrect'][i] + year
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig
 output['Years to start CA'] = StartCA

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
compnames[1]
comps
compnames
def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))

 StartCA = [None] * len(comps)
 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  StartCA[i] = compnames[i] + comps[i]['StartCorrect'] + year
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig
 output['Years to start CA'] = StartCA

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
compnames[1]
comps[1]['StartCorrect']
compnames[1] = comps[1]['StartCorrect']
compnames[1] + comps[1]['StartCorrect']
compnames = ["TSGHA", "SSTVC","TSEM"]
compnames[1] + comps[1]['StartCorrect']
compnames[1] + str((comps[1]['StartCorrect']+year))
year = 2022
compnames[1] + str((comps[1]['StartCorrect']+year))
compnames[1] + str(( comps[1]['StartCorrect']+year))
def CU(comps, compnames):
 if len(compnames) != len(comps):
  return 'compnames and comps are of different length'
 if len(comps)<2:
  return 'Must input two or more components. The input comps should be a list of length two or more'
 from datetime import date
 today = date.today()
 year = int(today.strftime("%Y"))
 yearcheck = np.zeros(len(comps))
 repaircheck = np.zeros(len(comps))
 for i in range(0, len(comps)):
  yearcheck[i] = comps[i]['outyears']
  repaircheck[i] = len(comps[i]['number of repairs'])
 if min(yearcheck) != max(yearcheck):
  return 'Number of outyears needs to be the same for all components'
 if min(repaircheck) != max(repaircheck):
  return(print('Some components have been simulated under', int(max(repaircheck)), "Corrective Action scenarios while others have only been simulated under", int(min(repaircheck)), "Corrective Action scenarios. All components should be simulated under the same number of corrective action scenarios"))
 repairslen = min(repaircheck)


 R_Comb = np.empty(shape = (outyears,len(numrepairs), len(comps)))

 StartCA = [None] * len(comps)
 for i in range(0, len(comps)):
  newkeys = list(comps[i])
  StartCA[i] = compnames[i] + str((comps[i]['StartCorrect'] + year))
  for j in range(0, len(numrepairs)):
   R_Comb[:,j,i] = comps[i][newkeys[j]] 

 RComb = 1- R_Comb

 CUmat = 1
 for i in range(0, len(comps)):
  CUmat = CUmat * (RComb[:,:,i])
  #this is system unreliability  
 CUmat = 1- CUmat


 output = {}
 y = 0
 keys = [None] * len(numrepairs)
 for i in range(0,len(numrepairs)): 
  keys[y] = "Scenario="+str(y+1)
  y = y+1
 for i in range(0,len(numrepairs)): 
  output[keys[i]] = CUmat[:,i]

 df = np.empty(shape = (len(numrepairs), len(comps)))
 row_names = [[]]* len(numrepairs)
 for i in range(0, len(comps)):
  df[:,i] = comps[i]['number of repairs']
 for i in range(0, len(numrepairs)): 
  row_names[i] = "Scenario="+str(i)
 from matplotlib.pyplot import cm  
 df = pd.DataFrame(df, columns=compnames, index=row_names) 
 n = len(comps)
 c = [[]]*n
 color = iter(cm.rainbow(np.linspace(0, 1, n)))
 for i in range(n):
    c[i] = next(color)

 x = range(year,outyears+year)
 import matplotlib as mpl
 import matplotlib.pyplot as plt
 fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
 for i in range(0,len(numrepairs)):
  y = "Scenario="+str(i+1)
  ax.plot(x, output[y], label=y)
 ax.set_xlabel('Year')  # Add an x-label to the axes.
 ax.set_ylabel('Unreliability')  # Add a y-label to the axes.
 ax.set_title("Combined Unreliability")  # Add a title to the axes.
 for i in range(0,len(comps)):
  y = comps[i]['StartCorrect'] + year
  b = compnames[i] 
  plt.axvline(x=y,color = c[i], label = b, linestyle='--' )

 if len(numrepairs)+len(comps) < 10:
  ax.legend(bbox_to_anchor =(0.0, -0.2));


 output['Repair Scenarios']  = df    
 output['Combined Unreliability'] = fig
 output['Years to start CA'] = StartCA

 return(output)  



compnames = ["TSGHA", "SSTVC","TSEM"]
f = CU([test,test2,test3], compnames)
f
from git import Repo
pip install gitpython

## ---(Thu Jul 21 10:28:58 2022)---
python -m ensurepip
!git init
!git commit -m "first commit"
!git branch -M 'main'

## ---(Thu Jul 21 11:19:02 2022)---
!git init
!git commit -m "first commit"
!git branch -M 'main'
!git remote add origin https://github.com/silincai12/MARCRISK.git
!git push -u origin 'main'
!git push -u origin 'master'
!git push -u origin 'main'
!git push -f origin 'main'
cd
!git remote add origin https://github.com/silincai12/MARCRISK.git
git branch -M 'main'
!git branch -M 'main'
!git push -u origin 'main'
git add .
!git commit -m "initial commit"
!git push origin main
!git add temp.py
!git add Risk_tool_python3.py
cd
git commit -m "first commit"
git branch -M 'main'
git remote add origin https://github.com/silincai12/MARCRISK.git
git push -u origin 'main
!git commit -m "first commit"
!git branch -M 'main'
!git remote add origin https://github.com/silincai12/MARCRISK.git
!git push -u origin 'main
cd 
cd .spyder-y3
cd C;/Users/dummy account/.spyder-y3
cd C:\Users\dummy account\.spyder-y3
cd C:\Users\dummy account\.spyder-py3
cd
import os

os.chdir(C:\Users\dummy account\.spyder-py3)
import os

os.chdir("C:\Users\dummy account\.spyder-py3")
cd C:\Users\dummy account\.spyder-py3
!git init
!git add README.md
!git commit -m "first commit"
!git branch -M 'main'
!git remote add origin https://github.com/silincai12/MARCRISK.git
!git push -u origin 'main'

!git branch -M 'main'
git push -u origin 'main'
!git push -u origin 'main'
!git push -u origin 'mmaster'
!git push -u origin 'master'
!git push -u origin 'main'
!git show-ref
!git push origin HEAD:<branch>
!git add .
!git commit -m "my first commit"
!git push -u origin master
!git push origin main
!git show-ref
!git init 
!git status
!git add -A
!git commit -f "first commit"
!git branch 'main'
!git remote add origin https://github.com/silincai12/MARCRISK.git
!git push -f origin 'main'
!git add Risk_tool_python3.py
!git commit -f "MARC"
!git commit "MARC"
!git push -f origin 'main'
mkdir newrepo && cd newrepo
git remote add origin https://github.com/silincai12/testestes.git
git add .
git commit -m "my first commit"
git push -u origin master
!mkdir newrepo && cd newrepo
!git remote add origin https://github.com/silincai12/testestes.git
!git add .
!git commit -m "my first commit"
!git push -u origin master