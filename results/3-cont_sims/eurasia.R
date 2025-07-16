library(slendr)
library(ggplot2)

map <- world(xrange = c(-13, 70), yrange = c(18, 65), crs = 3035)

R1 <- region(
  "EHG range", map,
  polygon = list(c(26, 55), c(38, 53), c(48, 53), c(60, 53),
                 c(60, 60), c(48, 63), c(38, 63), c(26, 60))
)
R2 <- region(
  "Europe", map,
  polygon = list(
    c(-8, 35), c(-5, 36), c(10, 38), c(20, 35), c(25, 35),
    c(33, 45), c(20, 58), c(-5, 60), c(-15, 50)
  )
)
R3 <- region(
  "Anatolia", map,
  polygon = list(c(28, 35), c(40, 35), c(42, 40),
                 c(30, 43), c(27, 40), c(25, 38))
)
R4 <- join(R2, R3)
R5 <- region(
  "YAM range", map,
  polygon = list(c(26, 50), c(38, 49), c(48, 50),
                 c(48, 56), c(38, 59), c(26, 56))
)

ooa_trajectory <- list(c(40, 30), c(50, 30), c(60, 40), c(45, 55))


map <- world(xrange = c(-13, 70), yrange = c(18, 65), crs = 3035)

ooa <- population(
  "OOA", time = 50000, N = 500, remove = 23000,
  map = map, center = c(33, 30), radius = 400e3
) %>%
  move(trajectory = ooa_trajectory, start = 50000, end = 40000, snapshots = 30)

ehg <- population(
  "EHG", time = 28000, N = 1000, parent = ooa, remove = 6000,
  map = map, polygon = R1
)

eur <- population(
  "EUR", time = 30000, N = 2000, parent = ooa,
  map = map, polygon = R2
) %>%
  resize(N = 10000, time = 5000, end = 0, how = "exponential")

ana <- population(
  "ANA", time = 25000, N = 4000, parent = ooa, remove = 3000,
  map = map, polygon = R3
) %>%
  expand_range(by = 3e6, start = 10000, end = 7000, polygon = R4, snapshots = 15)

yam <- population(
  "YAM", time = 7000, N = 600, parent = ehg, remove = 2500,
  map = map, polygon = R5
) %>%
  move(trajectory = list(c(15, 50)), start = 5000, end = 3000, snapshots = 10)

gf <- list(
  gene_flow(ana, to = yam, rate = 0.5, start = 6500, end = 5000),
  gene_flow(ana, to = eur, rate = 0.6, start = 8000, end = 6000),
  gene_flow(yam, to = eur, rate = 0.7, start = 3500, end = 3000)
)

model <- compile_model(
  populations = list(ooa, ehg, eur, ana, yam), gene_flow = gf,
  generation_time = 30, resolution = 10e3,
  competition = 150e3, mating = 120e3, dispersal = 90e3
)

samples <- schedule_sampling(
  model, times = seq(0, 50000, by = 1000),
  list(ehg, 20), list(ana, 20), list(yam, 20), list(eur, 20)
)

# samples <- schedule_sampling(
#   model, times = c(0),
#   list(ehg, 20), list(ana, 20), list(yam, 20), list(eur, 20)
# )

plot_model(model, sizes = FALSE)
plot_map(model)

ts <- slim(
  model, burnin = 200000, samples = samples, random_seed = SEED,
  sequence_length = 200000, recombination_rate = 1e-8
)