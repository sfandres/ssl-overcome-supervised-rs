# import os

# times = 2
# queryIndex = 129
# exp = [2, 29, 400]
# exp = exp[0]
# # cpuBatchSizes = range(1000,30000,1000)
# cpuBatchSizes = [29257]
# # gpuBatchSizes = range(100,1500,100)
# gpuBatchSizes = [1200]
# # num_threads = [1,2,4,6]
# num_threads = [1]

# # compile
# # os.system("source Launch.sh" + " g++" + " O3" + " openmp")

# print ("Experiments: %d " % times)
# for s in range(0,times):
#         for c in cpuBatchSizes:
#                 print ("CPU batch: %d " % (c))
#                 for g in gpuBatchSizes:
#                         print ("GPU batch: %d " % (g))
#                         for t in num_threads:
#                                 print ("Threads: %d " % (t))
#                                 os.system("./dmcCUDA {} {} {} {} {} {} {} {}".format("LI",t,c,g,queryIndex,"/home/ajsanchez/Documents/Fingerprints/RNew.txt",
#                                 "/home/ajsanchez/Documents/Fingerprints/LatentNew.txt","/home/ajsanchez/Documents/Fingerprints/Impressions" + str(exp) + "kNew.txt"))