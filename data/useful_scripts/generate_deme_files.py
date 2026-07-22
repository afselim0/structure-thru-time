#%%
import msprime
import demes
import stdpopsim
import demesdraw
import numpy as np

main_demes_dir_1 = "../demes/main_1/"
main_demes_dir_2 = "../demes/main_2/"
supp_demes_dir = "../demes/supp/"
stdpopsim_demes_dir = "../demes/stdpopsim/"
#%%
##--------------------------------------------------------
## Demogaphy functions
##--------------------------------------------------------
def pop_split(N0,N1,N2,t_split,t_mig_start=0,t_mig_end=0,m=0):
    """Creates a demography object of a population split that gave rise to 2 subpopulations, with no outgroup

    Args:
        N (int): effective population size of each subpopulation
        n (int): number of sampled individuals from each subpopulation
        t_split (float): time since the split
        t_mig (float): start time of a migration pulse
        mig_duration (float): duration of the migration pulse
        m (float): migration rate during the pulse

    Returns:
        tskit.demography: a demography object with the specified population history; 2 splits, a recent one, and a more distant one that defines an outgroup
        dict: number of samples individuals from each subpopulation 
    """
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="Ancestral")

    if m!=0:
        assert t_mig_start!=0 and t_mig_end!=0
        demography.add_symmetric_migration_rate_change(time=t_mig_start,populations=['A','B'],rate=m)
        demography.add_symmetric_migration_rate_change(time=t_mig_end,populations=['A','B'],rate=0)

    demography.sort_events()
    
    return demography

def split_sawtooth(N0, t_split):
    sawtooth_model = stdpopsim.get_species("HomSap").get_demographic_model("Zigzag_1S14").model
    model_dict = sawtooth_model.asdict()
    deepest_split_time = model_dict["events"][-1]["time"]
    assert t_split > deepest_split_time, f"t_split must be greater than {deepest_split_time}"

    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size= N0)

    demography.add_population(name="A", initial_size= N0, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N0, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="Ancestral")
    for event in model_dict["events"]:
        demography.add_population_parameters_change(time=event["time"], initial_size=event["initial_size"], growth_rate=event["growth_rate"], population = "A")
        demography.add_population_parameters_change(time=event["time"], initial_size=event["initial_size"], growth_rate=event["growth_rate"], population = "B")

    demography.sort_events()
    
    return demography

def pop_split_admix(N0,N1,N2,N_admix,t_split,t_admix,admix_prop=0.5):
    
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N_admix, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="Ancestral")

    demography.add_admixture(time=t_admix,derived="C",ancestral=["A","B"],proportions=[admix_prop,1-admix_prop])

    demography.sort_events()
    
    return demography

def pop_split_2(N_dict,t_split_1,t_split_2):
    
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size= N_dict["Ancestral"])

    demography.add_population(name="A", initial_size= N_dict["A"], default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N_dict["B"], default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N_dict["C"], default_sampling_time = 0)
    demography.add_population(name="temp_pop", initial_size= N_dict["temp_pop"])

    demography.add_population_split(time=t_split_1+t_split_2, derived=["temp_pop", "C"], ancestral="Ancestral")
    demography.add_population_split(time=t_split_1, derived=["A", "B"], ancestral="temp_pop")

    demography.sort_events()
    
    return demography

def recent_mig_change(N,t_split,t_change,m1,m2):
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size= N)

    demography.add_population(name="A", initial_size= N, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B", "C"], ancestral="Ancestral")

    demography.set_symmetric_migration_rate(populations= ["B","C"],rate= m1)
    demography.add_symmetric_migration_rate_change(time= t_change, populations= ["B","C"], rate= 0)
    demography.add_symmetric_migration_rate_change(time= t_change, populations= ["B","A"], rate= m2)

    demography.sort_events()

    return demography

def anomaly_zone_4(Ne, AB_split_time, distant_split_branch_length):
    # For a species tree composed of 4 species, with topology ((A,B),C),D), where D is the outgroup, and C is a distant branch, the anomalous zone is defined by the following inequality:
    # recent_split_branch_length < b(x), where x = distant_split_branch_length
    # b(x) = log(2/3 + \frac{5e^{2x}-2}{6(3e^{2x}-2e^{2x})}) \cite{DegnanRosenberg2006}

    b = lambda x: np.log(2/3 + (5*np.exp(2*x)-2)/(6*(3*np.exp(2*x)-2*np.exp(x))))
    recent_split_branch_length = (b(distant_split_branch_length/Ne) / 2)*Ne

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="D", initial_size= Ne, default_sampling_time = 0)

    demography.add_population(name="AB", initial_size= Ne)
    demography.add_population(name="ABC", initial_size= Ne)
    demography.add_population(name="ABCD", initial_size= Ne)

    demography.add_population_split(time=AB_split_time, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=AB_split_time + recent_split_branch_length, derived=["AB", "C"], ancestral="ABC")
    demography.add_population_split(time=AB_split_time + recent_split_branch_length + distant_split_branch_length, derived=["ABC", "D"], ancestral="ABCD")

    demography.sort_events()

    return demography

def complex_4_pops(Ne, Ne_bottleneck, Ne_A_recent, Ne_B_recent, Ne_C_recent, C_recent_growth_stop_time, A_B_split_time, AB_C_split_time, ABC_D_split_time):
    # b = lambda x: np.log(2/3 + (5*np.exp(2*x)-2)/(6*(3*np.exp(2*x)-2*np.exp(x))))
    # recent_split_branch_length = (b(distant_split_branch_length/Ne) / 2)*Ne
    # AB_C_split_time = A_B_split_time + recent_split_branch_length
    # ABC_D_split_time = AB_C_split_time + distant_split_branch_length

    growth_rate_A = (1/A_B_split_time)*np.log(Ne_A_recent/Ne_bottleneck)
    growth_rate_B = (1/A_B_split_time)*np.log(Ne_B_recent/Ne_bottleneck)


    C_growth_timepoints = np.logspace(np.log10(C_recent_growth_stop_time),np.log10(AB_C_split_time), num=4)
    growth_rate_C_1 = 1/(C_growth_timepoints[1]-C_growth_timepoints[0])*np.log(Ne_C_recent/Ne_bottleneck)
    growth_rate_C_2 = 1/(C_growth_timepoints[2]-C_growth_timepoints[1])*np.log(Ne_bottleneck/Ne_C_recent)
    growth_rate_C_3 = 1/(C_growth_timepoints[3]-C_growth_timepoints[2])*np.log(Ne_C_recent/Ne_bottleneck)

    demography = msprime.Demography()

    demography.add_population(name="A", initial_size= Ne_A_recent, growth_rate=growth_rate_A, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= Ne_B_recent, growth_rate=growth_rate_B, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= Ne_C_recent, growth_rate=0, default_sampling_time = 0)
    demography.add_population(name="D", initial_size= Ne, growth_rate=0, default_sampling_time = 0)

    demography.add_population(name="AB", initial_size= Ne_bottleneck)
    demography.add_population(name="ABC", initial_size= Ne_bottleneck)
    demography.add_population(name="ABCD", initial_size= Ne)

    demography.add_population_split(time=A_B_split_time, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=AB_C_split_time, derived=["AB", "C"], ancestral="ABC")
    demography.add_population_split(time=ABC_D_split_time, derived=["ABC", "D"], ancestral="ABCD")

    demography.add_population_parameters_change(time=C_growth_timepoints[0], population="C", growth_rate=growth_rate_C_1)
    demography.add_population_parameters_change(time=C_growth_timepoints[1], population="C", growth_rate=growth_rate_C_2)
    demography.add_population_parameters_change(time=C_growth_timepoints[2], population="C", growth_rate=growth_rate_C_3)


    demography.sort_events()
    return demography

def coor(index,K):
    x=index%K
    y=index//K
    return x,y

def Stepping_Stones_1d(N,K,m):
    demography = msprime.Demography.stepping_stone_model([N]*K, m/2, boundaries=True)
    return demography

def Stepping_Stones_2d(N,K,m):
    #N: Ne of each deme
    #K: Number of demes per side (total number of demes is K*K)
    #m: probability of obtaining the new individual from a nearby deme (backward migration rate)
    #n: sample size drawn from each deme
    deme_names = [f"x{row}y{col}" for col, row in (coor(i, K) for i in range(K**2))]
    demography = msprime.Demography()
    for name in deme_names:
        demography.add_population(name=name, initial_size=N, default_sampling_time=0)
    for i in range(K**2):
        for j in range(K**2):
            x1,y1 = coor(i,K)
            x2,y2 = coor(j,K)
            if (abs(x1-x2)==1 and abs(y1-y2)==0) or (abs(y1-y2)==1 and abs(x1-x2)==0):
                demography.set_migration_rate(source=deme_names[i], dest=deme_names[j], rate=m/4)
    return demography

def Multi_Stepping_Stones_2d(N, K, m_within, m_between, split_time, n_groups=2):
    #N: Ne of each deme (and of the ancestral population)
    #K: Number of demes per side within a group (each group has K*K demes, total demes is K*K*n_groups)
    #m_within: backward migration rate between adjacent demes within the same group's 2D stepping-stone grid
    #m_between: backward migration rate between every pair of demes belonging to different groups
    #split_time: time in the past at which the single ancestral population splits into all K*K*n_groups demes
    demes_per_group = K**2
    n_demes = demes_per_group*n_groups
    # name each deme by its group and its (row, column) position within the group's KxK grid
    deme_names = [f"g{g}_{row}_{col}" for g in range(n_groups) for col,row in (coor(i,K) for i in range(demes_per_group))]

    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size=N*n_demes)
    for name in deme_names:
        demography.add_population(name=name, initial_size=N, default_sampling_time=0)
    demography.add_population_split(time=split_time, derived=deme_names, ancestral="Ancestral")

    # within-group stepping-stone migration: adjacent demes (in the group's own KxK grid) exchange migrants at m_within
    for g in range(n_groups):
        offset = g*demes_per_group
        for i in range(demes_per_group):
            for j in range(demes_per_group):
                x1,y1 = coor(i,K)
                x2,y2 = coor(j,K)
                if (abs(x1-x2)==1 and abs(y1-y2)==0) or (abs(y1-y2)==1 and abs(x1-x2)==0):
                    demography.set_migration_rate(source=deme_names[offset+i], dest=deme_names[offset+j], rate=m_within[g])

    # between-group migration: every deme pair belonging to different groups exchanges migrants at m_between
    for gi in range(n_groups):
        for gj in range(n_groups):
            if gi==gj:
                continue
            for i in range(demes_per_group):
                for j in range(demes_per_group):
                    demography.set_migration_rate(source=deme_names[gi*demes_per_group+i], dest=deme_names[gj*demes_per_group+j], rate=m_between)

    demography.sort_events()
    return demography

def two_admix(Ne, deep_split_time, recent_split_time, admix_time_1, admix_time_2,admix_prop_1=0.5, admix_prop_2=0.5):
    
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size= Ne)
    demography.add_population(name="AB_ancestor", initial_size= Ne)
    demography.add_population(name="CD_ancestor", initial_size= Ne)

    demography.add_population(name="A", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="D", initial_size= Ne, default_sampling_time = 0)

    demography.add_population(name="Admix_1", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="Admix_2", initial_size= Ne, default_sampling_time = 0)

    demography.add_population_split(time=deep_split_time, derived=["AB_ancestor", "CD_ancestor"], ancestral="Ancestral")
    demography.add_population_split(time=recent_split_time, derived=["A", "B"], ancestral="AB_ancestor")
    demography.add_population_split(time=recent_split_time, derived=["C", "D"], ancestral="CD_ancestor")

    demography.add_admixture(time=admix_time_1,derived="Admix_1",ancestral=["A","B"],proportions=[admix_prop_1,1-admix_prop_1])
    demography.add_admixture(time=admix_time_2,derived="Admix_2",ancestral=["C","D"],proportions=[admix_prop_2,1-admix_prop_2])

    demography.sort_events()
    
    return demography

#%%
Ne=1e4

# 1-split
N0 = 2*Ne
N1 = 1*Ne
N2 = 0.5*Ne
t_split = Ne / 5
demography = pop_split(N0,N1,N2,t_split)
graph = demography.to_demes()
demes.dump(graph, main_demes_dir_1 + "split.yaml")

# 2-split_mig
Nm = 0.1

N0 = 2*Ne
N1 = 1*Ne
N2 = 0.5*Ne
t_split = Ne / 5
t_mig = Ne/2
mig_duration = Ne/2
m = Nm / Ne

demography = pop_split(N0,N1,N2,t_split,t_mig,mig_duration,m)
graph = demography.to_demes()

demes.dump(graph, main_demes_dir_1 + f"split_mig_Nm_{"".join(str(Nm).split("."))}.yaml")

# 3- split_sawtooth
demography = split_sawtooth(Ne,t_split=40000)
graph = demography.to_demes()
demes.dump(graph, main_demes_dir_1 + "split_sawtooth.yaml")

# 4- mig_change
t_split = 2000
t_change = 200
m1 = 1e-4
m2 = 1e-3

demography = recent_mig_change(Ne,t_split,t_change,m1,m2)
graph = demography.to_demes()
demes.dump(graph, main_demes_dir_1 + "mig_change.yaml")

# 5- Anomaly Zone
Ne = 1e4
AB_split_time = Ne/10
distant_split_branch_length = Ne/10
demography = anomaly_zone_4(Ne, AB_split_time, distant_split_branch_length)
graph = demography.to_demes()
demes.dump(graph, main_demes_dir_1 + "anomaly_zone.yaml")
#%%
# 6- Complex 4 pops
Ne = 1.5e4
Ne_bottleneck = Ne/10
A_B_split_time = 1000
AB_C_split_time = 1200
ABC_D_split_time = 1400

Ne_A_recent = 2*Ne
Ne_B_recent = 4*Ne
Ne_C_recent = 4*Ne
C_recent_growth_stop_time = 30

demography = complex_4_pops(Ne, Ne_bottleneck, Ne_A_recent, Ne_B_recent, Ne_C_recent, C_recent_growth_stop_time, A_B_split_time, AB_C_split_time, ABC_D_split_time)
demes.dump(demography.to_demes(), main_demes_dir_1 + "complex_4_pops.yaml")
# %%
# 7- Stepping Stones 2D
N = 100
m = 0.0001
K = 10
demography = Stepping_Stones_2d(N,K,m)
graph = demography.to_demes()
demes.dump(graph, main_demes_dir_1 + "stepping_stones_2d.yaml")
# %%
# 8- Two Admix
Ne = 1e4
deep_split_time = 4000
recent_split_time = 2000
admix_time_1 = 500
admix_time_2 = 100

demography = two_admix(Ne, deep_split_time, recent_split_time, admix_time_1, admix_time_2,admix_prop_1=0.5, admix_prop_2=0.5)
graph = demography.to_demes()
demes.dump(graph, main_demes_dir_1 + "two_admix.yaml")
#%%
# Double 2d SS model
N = 1e2
m_within = [1e-2, 5e-2]
m_between = 1e-6
K=5
split_time = 2500
demography = Multi_Stepping_Stones_2d(N, K, m_within, m_between, split_time, n_groups=2)
graph = demography.to_demes()
# simplified=False: demes' YAML simplification tries to collapse migration rates into
# symmetric groups via an all-subsets search (itertools.combinations over every deme
# sharing a rate), which is combinatorial and hangs once m_between links many demes
demes.dump(graph, main_demes_dir_1 + "double_2d_SS.yaml", simplified=False)
# %%
# SS demes
import os
SS_demes_dir_1 = "../demes/SS_demes/"
os.makedirs(SS_demes_dir_1, exist_ok=True)
Ne = 100
K1 = 100
K2 = 10
Nm_list = [5,1,0.1]
m_list = [Nm/Ne for Nm in Nm_list]
for m in m_list:
    demography = Stepping_Stones_1d(Ne,K1,m)
    graph = demography.to_demes()
    m_str = str(m).split(".")[1]
    demes.dump(graph, SS_demes_dir_1 + f"SS_1d_N{Ne}_m{m_str}.yaml", simplified=False)

    demography = Stepping_Stones_2d(Ne,K2,m)
    graph = demography.to_demes()
    demes.dump(graph, SS_demes_dir_1 + f"SS_2d_N{Ne}_m{m_str}.yaml", simplified=False)
# %%
