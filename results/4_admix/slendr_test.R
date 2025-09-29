library(slendr)
init_env()
SEED=42

world_size=1000
N=1000
pop_time = 1000
resolution = 1
competition = 10
dispersal = 20
mating = 10
recombination_rate = 1e-8
sequence_length = 100e6

# Defining a blank map with a square grid
map = world(xrange = c(0, world_size), yrange = c(0, world_size), landscape = "blank")

# A single population that spans the whole map
pop = population(
  name="main",
  time=pop_time,
  N=N,
  map=map,
  polygon = list(c(0, 0), c(world_size, 0), c(world_size, world_size), c(0, world_size)))

#plot_map(pop)

# Defining the model, with the main releveant parameter here being dispersal
model = compile_model(
  populations = list(pop), 
  generation_time = 1,
  resolution = resolution,
  competition = competition,
  mating = mating, 
  dispersal = dispersal,
  direction="backward"
)
ts <- slim(model, sequence_length = sequence_length, recombination_rate = recombination_rate, random_seed = SEED, locations = "locations")


# Sample n individuals from present time in the end
# n = 500
# samples = schedule_sampling(model, times = c(0), list(pop, n))

# ts <- slim(model, sequence_length = sequence_length, recombination_rate = recombination_rate, random_seed = SEED, ts=TRUE, samples=samples, locations = "locations")
# ts = ts_recapitate(ts,recombination_rate=recombination_rate,Ne=N)
# ts = ts_simplify(ts)

# ts_write(ts,"plane_test.trees")