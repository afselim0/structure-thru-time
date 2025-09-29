#%%
import os
import msprime
import demes
import stdpopsim
main_demes_dir = "../../data/main_deme_files/"
supp_demes_dir = "../../data/supp_deme_files/"
stdpopsim_demes_dir = "../../data/stdpopsim_deme_files/"

species = stdpopsim.get_species(id="HomSap")
#%%
def pop_split(N0,N1,N2,t_split,t_mig=0,mig_duration=0,m=0):
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
    demography.add_population(name="main", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="main")

    if m!=0:
        assert t_mig!=0 and mig_duration!=0
        demography.add_symmetric_migration_rate_change(time=t_mig,populations=['A','B'],rate=m)
        demography.add_symmetric_migration_rate_change(time=t_mig+mig_duration,populations=['A','B'],rate=0)

    demography.sort_events()
    
    return demography

def pop_split_admix(N0,N1,N2,N_admix,t_split,t_admix,admix_prop=0.5):
    
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N_admix, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="main")

    demography.add_admixture(time=t_admix,derived="C",ancestral=["A","B"],proportions=[admix_prop,1-admix_prop])

    demography.sort_events()
    
    return demography

def pop_split_2(N_dict,t_split_1,t_split_2):
    
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N_dict["main"])

    demography.add_population(name="A", initial_size= N_dict["A"], default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N_dict["B"], default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N_dict["C"], default_sampling_time = 0)
    demography.add_population(name="temp_pop", initial_size= N_dict["temp_pop"])

    demography.add_population_split(time=t_split_1+t_split_2, derived=["temp_pop", "C"], ancestral="main")
    demography.add_population_split(time=t_split_1, derived=["A", "B"], ancestral="temp_pop")

    demography.sort_events()
    
    return demography

# def recent_mig_change(N,t_split_1,t_split_2,t_change,m1,m2):
def recent_mig_change(N,t_split,t_change,m1,m2):
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N)

    demography.add_population(name="A", initial_size= N, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= N, default_sampling_time = 0)
    # demography.add_population(name="temp_pop", initial_size= N)

    # demography.add_population_split(time=t_split_2, derived=["temp_pop", "C"], ancestral="main")
    # demography.add_population_split(time=t_split_1, derived=["A", "B"], ancestral="temp_pop")
    demography.add_population_split(time=t_split, derived=["A", "B", "C"], ancestral="main")

    demography.set_symmetric_migration_rate(populations= ["B","C"],rate= m1)
    demography.add_symmetric_migration_rate_change(time= t_change, populations= ["B","C"], rate= 0)
    demography.add_symmetric_migration_rate_change(time= t_change, populations= ["B","A"], rate= m2)

    demography.sort_events()

    return demography

def SS(N,m,K):
    demography = demography = msprime.Demography()
    demography = demography.stepping_stone_model([N]*K, migration_rate=m, boundaries=True)
    for pop in demography.populations:
        pop.default_sampling_time = 0
    return demography

def coor(index,K):
    x=index%K
    y=index//K
    return x,y

def SS_2d(N,m,K):
    #N: Number of individuals per deme
    #d: Number of demes per side (total number of demes is d*d)
    #m: probability of obtaining the new individual from a nearby deme (backward migration rate)
    #n: sample size drawn from each deme
    demography = msprime.Demography.isolated_model(initial_size = [N]*(K**2))

    M = demography.migration_matrix
    for i in range(K**2):
        for j in range(K**2):
            x1,y1 = coor(i,K)
            x2,y2 = coor(j,K)
            if (abs(x1-x2)==1 and abs(y1-y2)==0) or (abs(y1-y2)==1 and abs(x1-x2)==0):
                M[i,j] = m/4
    
    for pop in demography.populations:
        pop.default_sampling_time = 0
    return demography
#%%
# Main examples

### 1-zigzag
model = species.get_demographic_model("Zigzag_1S14")
demography = model.model
graph = demography.to_demes()
demes.dump(graph,os.path.join(main_demes_dir,"zigzag.yaml"))

### 2-split
Ne = 1e4
N0=2*Ne
N1=1*Ne
N2=0.5*Ne
t_split = N2
demography = pop_split(N0,N1,N2,t_split)
graph = demography.to_demes()
demes.dump(graph,os.path.join(main_demes_dir,"split.yaml"))

### 3-split_mig
Ne = 1e4
N0=2*Ne
N1=1*Ne
N2=0.5*Ne
t_split = Ne/2
t_mig = Ne/4
mig_duration = Ne/4
m=1e-4
demography = pop_split(N0,N1,N2,t_split,t_mig,mig_duration,m)
graph = demography.to_demes()
demes.dump(graph,os.path.join(main_demes_dir,"split_mig_Nm_1.yaml"))

### 4-mig_change
t_change = 0.1*Ne
t_split = Ne/2
m1 = 1e-3 # Recent migration
m2 = 1e-1 # Early migration

demography = recent_mig_change(Ne,t_split,t_change,m1,m2)
graph = demography.to_demes()
demes.dump(graph, os.path.join(main_demes_dir,"mig_change.yaml"))

### 5-SS
# Ne = 100
# m=1e-3
# K=100
# demography = SS(Ne,m,K)
# graph = demography.to_demes()
# demes.dump(graph, os.path.join(main_demes_dir,"SS.yaml"))

#%%
models = species.demographic_models
for model in models:
    name = model.id
    demography = model.model
    graph = demography.to_demes()
    demes.dump(graph, os.path.join(stdpopsim_demes_dir,f"{name}.yaml"))
# %%
def recent_bottleneck(Ne=20000,N_bottleneck=200,t_bottleneck=20,t_split_1=700,t_split_2=1700):
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= Ne)
    demography.add_population(name="temp_1", initial_size= Ne)
    demography.add_population(name="temp_2", initial_size= Ne)


    demography.add_population(name="A", initial_size= N_bottleneck, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="D", initial_size= Ne, default_sampling_time = 0)

    demography.add_population_split(time=t_split_2, derived=["D", "temp_1"], ancestral="main")
    demography.add_population_split(time=t_split_1, derived=["C", "temp_2"], ancestral="temp_1")
    demography.add_population_split(time=t_bottleneck, derived=["A", "B"], ancestral="temp_2")
    
    demography.sort_events()
    return demography

demography = recent_bottleneck()
graph = demography.to_demes()
demes.dump(graph, demes_dir + "recent_bottleneck.yaml")
# %%
def three_way_admixture(Ne=20000,t_admix = 15,t_split_1=700,t_split_2=1700,x_B=0.5,x_C=0.35,x_D=0.15):

    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= Ne)
    demography.add_population(name="temp_1", initial_size= Ne)


    demography.add_population(name="A", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="D", initial_size= Ne, default_sampling_time = 0)

    demography.add_population_split(time=t_split_2, derived=["D", "temp_1"], ancestral="main")
    demography.add_population_split(time=t_split_1, derived=["B", "C"], ancestral="temp_1")
    demography.add_admixture(time=t_admix,derived="A",ancestral=["B","C","D"],proportions=[x_B,x_C,x_D])

    demography.add_census(t_admix+1)
    demography.sort_events()
    return demography

t_admix = 15
demography = three_way_admixture()
graph = demography.to_demes()
demes.dump(graph, demes_dir + "three_way_admixture.yaml")
#%%
def ghost_admixture(Ne=20000,t_admix=200,t_split_1=700,t_split_2=1700,x_B=0.5,x_C=0.35,x_D=0.15):

    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= Ne)
    demography.add_population(name="temp_1", initial_size= Ne)
    demography.add_population(name="X", initial_size= Ne, initially_active = False)

    demography.add_population(name="A", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= Ne, default_sampling_time = 0)
    demography.add_population(name="D", initial_size= Ne, default_sampling_time = 0)

    demography.add_population_split(time=t_split_2, derived=["B","D", "temp_1"], ancestral="main")
    demography.add_population_split(time=t_split_1, derived=["X", "C"], ancestral="temp_1")
    demography.add_admixture(time=t_admix, derived="A",ancestral=["B","X"],proportions=[0.5,0.5])

    demography.add_census(t_admix+1)
    demography.sort_events()
    return demography


t_admix = 200
demography = ghost_admixture()
graph = demography.to_demes()
demes.dump(graph, demes_dir + "ghost_admixture.yaml")
# %%
def subsequent_splits(Ne=20000):
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= Ne)
    demography.add_population(name="temp_1", initial_size= Ne)
    demography.add_population(name="temp_2", initial_size= Ne)
    
    demography.add_population(name="A", initial_size= Ne)
    demography.add_population(name="B", initial_size= Ne)
    demography.add_population(name="C", initial_size= Ne)
    demography.add_population(name="D", initial_size= Ne)

    demography.add_population_split(time=Ne, derived=["A", "B"], ancestral="temp_2")
    demography.add_population_split(time=1.1*Ne, derived=["C", "temp_2"], ancestral="temp_1")
    demography.add_population_split(time=1.2*Ne, derived=["D", "temp_1"], ancestral="main")

    demography.sort_events()
    return demography


demography = subsequent_splits()
graph = demography.to_demes()
demes.dump(graph, demes_dir + "subsequent_splits.yaml")
# %%
