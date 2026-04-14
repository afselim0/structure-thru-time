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
## Ancestral Demes
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
# %%
