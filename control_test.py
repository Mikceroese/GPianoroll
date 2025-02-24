import os
from bopt_dict import BayesOptDict as bopt_dict
from musegan import used_pitch_classes, tonal_distance, sample_from_midi
import numpy as np
import scipy.stats as sp

import matplotlib.pyplot as plt

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Specify the main folder
main_folder = 'exps'

fig, axs = plt.subplots(ncols = 2, sharey=True)

cs = []
n_all_updates = []
avg_nms = []
med_nms = []
long_nms = []
avg_ls = []
med_ls = []
long_ls = []

#-------------------------------------------------------

initial_budget = 21
initial_stds = []
n_update_curve = []
n_update_curve_notequal = []
upcs_g = []
upcs_b = []
tds = []

#-------------------------------------------------------

# Loop through each subfolder in the main folder
for subfolder in sorted(os.listdir(main_folder)):
    subfolder_path = os.path.join(main_folder, subfolder)
    bopt_log_path = os.path.join(subfolder_path, 'musegan_bopt.txt')
    midi_path = os.path.join(subfolder_path, 'sample.mid')
    npz_path = os.path.join(subfolder_path, 'results.npz')

    bo_dict = bopt_dict(bopt_log_path)

    # if subject is in the control group:
    if "CS" in subfolder:
        control=True
        cs.append(True)
        (mid_X, mid_Y) = bo_dict.get_sample(63)
        bo_dict.remove_sample(63)
        bo_dict.add_sample(mid_X,[mid_Y],0)
        color = 'b'
        ax_i = 1
    else:
        control=False
        cs.append(False)
        (mid_X, mid_Y) = bo_dict.get_sample(21)
        bo_dict.remove_sample(21,42)
        bo_dict.add_sample(mid_X,[mid_Y],0)
        color = 'r'
        ax_i = 0

    scores = (1 - bo_dict.mY)

    plt.sca(axs[ax_i])
    #bo_dict.print_curve()

    max_s = scores[0]
    n_updates = -1
    n_updates_notequal = 0
    updates = [0]
    update_curve = []
    update_curve_notequal = []
    for i, s in enumerate(scores):
        if s >= max_s:
            n_updates+=1
            if s > max_s:
                n_updates_notequal+=1
            max_s = s
            updates.append(i)
        update_curve.append(n_updates)
        update_curve_notequal.append(n_updates_notequal)
    updates = np.array(updates)
    plt.plot(update_curve,color+',-')
    n_update_curve.append(np.array(update_curve))
    n_update_curve_notequal.append(np.array(update_curve_notequal))
    no_max_streaks = updates[1:]-updates[:-1]
    no_max_streaks = np.append(no_max_streaks,64-updates[-1])

    low_streaks = []
    streak = 0
    window_size = 3
    for i_w, s in enumerate(scores[window_size:]):

        i = i_w + window_size

        mid_score = (np.min(scores[:i+1]) + np.max(scores[:i+1]))/2

        i_score = np.mean(scores[i-window_size:i])

        if i_score <= mid_score:
            streak+=1
        else:
            if streak>0:
                low_streaks.append(streak)
            streak = 0
    if streak>0:
        low_streaks.append(streak)
    low_streaks = np.array(low_streaks)

    if control:
        sample = sample_from_midi(midi_path)
    else:
        sample = np.load(npz_path)
        sample =sample["sample"]
    upcs = used_pitch_classes(sample,verbose=False)
    td = tonal_distance(sample,verbose=False)

    print(subfolder,":")
    print("Number of max updates:",n_updates)
    n_all_updates.append(n_updates)
    print("Mean No-Max-Streak:",np.mean(no_max_streaks))
    avg_nms.append(np.mean(no_max_streaks))
    print("Median No-Max-Streak:",np.median(no_max_streaks))
    med_nms.append(np.median(no_max_streaks))
    print("Longest No-Max-Streak:",np.max(no_max_streaks))
    long_nms.append(np.max(no_max_streaks))
    print("Mean Low-Streak:",np.mean(low_streaks))
    avg_ls.append(np.mean(low_streaks))
    print("Median Low-Streak:",np.median(low_streaks))
    med_ls.append(np.median(low_streaks))
    print("Longest Low-Streak:",np.max(low_streaks))
    long_ls.append(np.max(low_streaks))
    upcs_g.append(upcs[0])
    print("UPC (Guitar):",upcs[0])
    upcs_b.append(upcs[1])
    print("UPC (Bass):",upcs[1])
    tds.append(td[0])
    print("TD:",td[0])
    print("...........................................................")

cs = np.array(cs)
n_all_updates = np.array(n_all_updates)
avg_nms = np.array(avg_nms)
med_nms = np.array(med_nms)
long_nms = np.array(long_nms)
avg_ls = np.array(avg_ls)
med_ls = np.array(med_ls)
long_ls = np.array(long_ls)

print("-----------------------------------------------------------")

print("CONTROL GROUP:")
print("Number of max updates:",np.mean(n_all_updates[cs==True]))
print("Avg No-Max-Streak:",np.mean(avg_nms[cs==True]))
print("Avg Median No-Max-Streak:",np.mean(med_nms[cs==True]))
print("Avg Longest No-Max-Streak:",np.mean(long_nms[cs==True]))
print("Avg Low-Streak:",np.mean(avg_ls[cs==True]))
print("Avg Median Low-Streak:",np.mean(med_ls[cs==True]))
print("Avg Longest Low-Streak:",np.mean(long_ls[cs==True]))

print("----------------------------------------------------------")

print("OPT GROUP:")
print("Number of max updates:",np.mean(n_all_updates[cs==False]))
print("Avg No-Max-Streak:",np.mean(avg_nms[cs==False]))
print("Avg Median No-Max-Streak:",np.mean(med_nms[cs==False]))
print("Avg Longest No-Max-Streak:",np.mean(long_nms[cs==False]))
print("Avg Low-Streak:",np.mean(avg_ls[cs==False]))
print("Avg Median Low-Streak:",np.mean(med_ls[cs==False]))
print("Avg Longest Low-Streak:",np.mean(long_ls[cs==False]))

# ----------------------------------------------------------------

fig, axs = plt.subplots(ncols=2,sharey='row')

num_runs = 8
n_update_curve = np.array(n_update_curve)
n_update_curve_notequal = np.array(n_update_curve_notequal)

res = n_update_curve[cs==False]
res_mean = np.mean(res,axis=0)
res_mean_1 = np.mean(res,axis=0)
res_std = np.std(res,axis=0)

it = range(res_mean.shape[0])
t_limits = sp.t.interval(0.95,num_runs) / np.sqrt(num_runs)

axs[0].plot(it,res_mean, 'r,-', linewidth=2, label="Bayesopt")
axs[0].fill(np.concatenate([it,it[::-1]]),
        np.concatenate([res_mean + t_limits[0] * res_std,
                        (res_mean + t_limits[1] * res_std)[::-1]]),
        alpha=.3, fc='r', ec='None')

res = n_update_curve_notequal[cs==False]
res_mean = np.mean(res,axis=0)
res_std = np.std(res,axis=0)

it = range(res_mean.shape[0])
t_limits = sp.t.interval(0.95,num_runs) / np.sqrt(num_runs)

axs[1].plot(it,res_mean, 'r,-', linewidth=2, label="Bayesopt")
axs[1].fill(np.concatenate([it,it[::-1]]),
        np.concatenate([res_mean + t_limits[0] * res_std,
                        (res_mean + t_limits[1] * res_std)[::-1]]),
        alpha=.3, fc='r', ec='None')

res = n_update_curve[cs==True]
res_mean = np.mean(res,axis=0)
res_std = np.std(res,axis=0)

it = range(res_mean.shape[0])
t_limits = sp.t.interval(0.95,num_runs) / np.sqrt(num_runs)

axs[0].plot(it,res_mean, 'b,-', linewidth=2, label="Random")
axs[0].fill(np.concatenate([it,it[::-1]]),
        np.concatenate([res_mean + t_limits[0] * res_std,
                        (res_mean + t_limits[1] * res_std)[::-1]]),
        alpha=.3, fc='b', ec='None')


res = n_update_curve_notequal[cs==True]
res_mean = np.mean(res,axis=0)
res_std = np.std(res,axis=0)

it = range(res_mean.shape[0])
t_limits = sp.t.interval(0.95,num_runs) / np.sqrt(num_runs)

axs[1].plot(it,res_mean, 'b,-', linewidth=2, label="Random")
axs[1].fill(np.concatenate([it,it[::-1]]),
        np.concatenate([res_mean + t_limits[0] * res_std,
                        (res_mean + t_limits[1] * res_std)[::-1]]),
        alpha=.3, fc='b', ec='None')

axs[0].set_xlabel("Iteration")
axs[1].set_xlabel("Iteration")
axs[0].set_ylabel("Total updates")
axs[1].set_ylabel("Total updates")
axs[0].set_title("Best score updates up to iteration (higher or equal)")
axs[1].set_title("Best score updates up to iteration (strictly higher)")
axs[1].legend()

print("----------------------------------------------------------")

fig, axs_all = plt.subplots(nrows=2, ncols=3,sharey='row',layout='tight')

upcs_g_bo = np.array(upcs_g)[cs==False]
upcs_g_r = np.array(upcs_g)[cs==True]
upcs_b_bo = np.array(upcs_b)[cs==False]
upcs_b_r = np.array(upcs_b)[cs==True]
tds_bo = np.array(tds)[cs==False]
tds_r = np.array(tds)[cs==True]

x = np.arange(8)
width = 0.3

axs = axs_all[0]

axs[0].set_title("TD (BayesOpt)")
axs[0].bar(x,tds_bo,width,label="Guitar-Bass")
axs[0].set_xticks(x,x+1)
axs[0].set_xlabel("Subject")
non_zero_td_bo = tds_bo[np.nonzero(tds_bo)]
print("Avg. TD (bo):",np.mean(non_zero_td_bo))
print("Std. TD (bo):",np.std(non_zero_td_bo))

axs[1].set_title("TD (Random)")
axs[1].bar(x,tds_r,width,label="Guitar-Bass")
axs[1].set_xticks(x,x+1)
axs[1].set_xlabel("Subject")
non_zero_td_r = tds_r[np.nonzero(tds_r)]
print("Avg. TD (r):",np.mean(non_zero_td_r))
print("Std. TD (r):",np.std(non_zero_td_r))

"""
axs[0].plot([-0.5,7.5],[np.mean(non_zero_td),np.mean(non_zero_td)],color="blue",marker=',',linestyle="-.",label="User mean")
axs[0].plot([-0.5,7.5],[np.mean(non_zero_td),np.mean(non_zero_td)],color="blue",marker=',',linestyle="-.",label="User mean")
axs[1].plot([-0.5,7.5],[np.mean(non_zero_td ),np.mean(non_zero_td )],color="blue",marker=',',linestyle="-.",label="User mean")
"""

axs[1].legend()

# ------------------------------------------------------------

axs = axs_all[1]

axs[0].set_title("UPCs (BayesOpt)")
axs[0].bar(x,upcs_g_bo,width,label="Guitar")
axs[0].bar(x+width,upcs_b_bo,width,label="Bass")
print("Avg. Guitar UPC (bo):",np.mean(upcs_g_bo))
print("Std. Guitar UPC (bo):",np.std(upcs_g_bo))
print("Avg. Bass UPC (bo):",np.mean(upcs_b_bo))
print("Std. Bass UPC (bo):",np.std(upcs_b_bo))
axs[0].set_xticks(x+width/2,x+1)
axs[0].set_xlabel("Subject")

axs[1].set_title("UPCs (Random)")
axs[1].bar(x,upcs_g_r,width,label="Guitar")
axs[1].bar(x+width,upcs_b_r,width,label="Bass")
print("Avg. Guitar UPC (r):",np.mean(upcs_g_r))
print("Std. Guitar UPC (r):",np.std(upcs_g_r))
print("Avg. Bass UPC (r):",np.mean(upcs_b_r))
print("Std. Bass UPC (r):",np.std(upcs_b_r))
axs[1].set_xticks(x+width/2,x+1)
axs[1].set_xlabel("Subject")

"""
axs[0].plot([-0.5,7.5],[3.22,3.22],",b--",label="Model mean (Guitar)")
axs[0].plot([-0.5,7.5],[2.01,2.01],color='orange',marker=',',linestyle='--',label="Model mean (Bass)")
axs[0].plot([-0.5,7.5],[np.mean(upcs_g_bo),np.mean(upcs_g_bo)],color="blue",marker=',',linestyle="..",label="Bayesopt mean (Guitar)")
axs[1].plot([-0.5,7.5],[np.mean(upcs_g_r),np.mean(upcs_g_r)],color="blue",marker=',',linestyle="-.",label="Random mean (Guitar)")
axs[0].plot([-0.5,7.5],[np.mean(upcs_b_bo),np.mean(upcs_b_bo)],color="orange",marker=',',linestyle="-.",label="Bayesopt mean (Bass)")
axs[1].plot([-0.5,7.5],[np.mean(upcs_b_r),np.mean(upcs_b_r)],color="orange",marker=',',linestyle="-.",label="Random mean (Bass)")
"""

axs[1].legend()

# -----------------------------------------------------------------

metrics = ["UPC G","UPC B","TD"]
groups = ["Dataset","Model","Bayesopt","Random"]

# Dataset, Model, Bayesopt group, Random group
# UPC_G, UPC_B, TD
# Values for the Dataset and Model are obtained from track_test.py
means = np.array([
    [3.43,3.37,np.mean(upcs_g_bo),np.mean(upcs_g_r)],
    [2.25,2.14,np.mean(upcs_b_bo),np.mean(upcs_b_r)],
    [1.28,1.32,np.mean(non_zero_td_bo),np.mean(non_zero_td_r)]
])
stds = np.array([
    [2.02,1.93,np.std(upcs_g_bo),np.std(upcs_g_r)],
    [1.11,0.90,np.std(upcs_b_bo),np.std(upcs_b_r)],
    [0.43,0.42,np.std(non_zero_td_bo),np.std(non_zero_td_r)]
])
ns = np.array([
    [57201],
    [250000],
    [8],
    [8]
])
t_limits = sp.t.interval(0.95,ns) / np.sqrt(ns)
stds = stds * t_limits[1].T

# Bar plot settings
bar_width = 0.2
x = np.arange(len(metrics)-1)  # X locations for the groups

# Create figure and axis
ax0 = axs_all[0][2]
ax1 = axs_all[1][2]

# Plot bars for each group
for i in range(len(groups)):
    ax1.bar(x + i * bar_width, means[:2, i], yerr=stds[:2, i], width=bar_width,
           label=groups[i], capsize=5, alpha=0.8)
    ax0.bar(i * bar_width, means[2, i], yerr=stds[2, i], width=bar_width,
           label=groups[i], capsize=5, alpha=0.8)

# Labels and title
ax0.set_title("Avg. TDs per group")
w = bar_width * (len(groups) / 2 - 0.5)
ax0.set_xticks([w])
ax0.set_xticklabels([metrics[2]])
ax0.set_xlim([-w,3*w])
ax1.set_title("Avg. UPCs per group")
ax1.set_xticks(x + bar_width * (len(groups) / 2 - 0.5))
ax1.set_xticklabels(metrics[:2])

# Legend
ax0.legend(loc='upper left',bbox_to_anchor=(1.2,1))

plt.show()