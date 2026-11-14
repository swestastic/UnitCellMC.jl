@testset "Proposal schemes" begin
    geometry = square_geometry(3)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    rng = Xoshiro(3)

    @test ucmc.MetropolisAlgorithm().proposal isa ucmc.RandomProposal
    @test ucmc.GlauberAlgorithm().proposal isa ucmc.RandomProposal
    @test ucmc.HeatbathAlgorithm().proposal isa ucmc.RandomProposal
    @test ucmc.OverrelaxationAlgorithm().proposal isa ucmc.RandomProposal
    @test ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()).proposal isa
        ucmc.SequentialProposal
    @test ucmc.GlauberAlgorithm(ucmc.SequentialProposal()).proposal isa
        ucmc.SequentialProposal
    @test ucmc.HeatbathAlgorithm(ucmc.SequentialProposal()).proposal isa
        ucmc.SequentialProposal

    # the default scheme carries no site-selection state; the Ising model's
    # discrete ΔE spectrum still gets a per-run acceptance cache
    state = ucmc.initialize_state(model, geometry; rng = rng)
    default_workspace = ucmc.algorithm_workspace(ucmc.MetropolisAlgorithm(), model, geometry, state)
    @test default_workspace isa ucmc.LocalUpdateWorkspace
    @test isnothing(default_workspace.proposal)
    @test default_workspace.acceptance isa ucmc.AcceptanceCache

    # continuous-spectrum models get no acceptance cache
    xy_model = ucmc.XYModel([1.0, 1.0], [0.0])
    xy_state = ucmc.initialize_state(xy_model, geometry; rng = rng)
    @test isnothing(ucmc.algorithm_workspace(
        ucmc.MetropolisAlgorithm(), xy_model, geometry, xy_state).acceptance)

    sequential_algorithms = (ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()),
                             ucmc.GlauberAlgorithm(ucmc.SequentialProposal()))

    for algorithm in sequential_algorithms
        workspace = ucmc.algorithm_workspace(algorithm, model, geometry, state)
        @test workspace isa ucmc.LocalUpdateWorkspace
        @test workspace.proposal isa ucmc.SequentialProposalWorkspace

        # sites are visited in site-index order, wrapping after a full sweep
        sites = [ucmc.propose(rng, algorithm, model, geometry, state; workspace = workspace).site
                 for _ in 1:geometry.n_sites]
        @test sites == collect(1:geometry.n_sites)
        @test ucmc.propose(rng, algorithm, model, geometry, state; workspace = workspace).site == 1

        # a stateful scheme without its workspace is an explicit error
        @test_throws ArgumentError ucmc.propose(rng, algorithm, model, geometry, state)
        @test_throws ArgumentError ucmc.step!(algorithm, model, geometry, state, 2.0; rng = rng)

        # running totals stay exact through sequential sweeps
        seq_state = ucmc.initialize_state(model, geometry; rng = rng)
        seq_workspace = ucmc.algorithm_workspace(algorithm, model, geometry, seq_state)
        for _ in 1:(3 * geometry.n_sites)
            ucmc.step!(algorithm, model, geometry, seq_state, 2.0;
                       rng = rng, workspace = seq_workspace)
            @test seq_state.energy ≈ spin_energy(model, geometry, seq_state.spins)
            @test seq_state.magnetization == sum(seq_state.spins)
        end
    end

    # the drivers build and thread the workspace automatically
    for algorithm in sequential_algorithms
        container = ucmc.MeasurementContainer(model, geometry, 4, 2;
                                              diagnostics = [:acceptance_ratio])
        parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
        ucmc.run_simulation(algorithm, model, geometry,
                            ucmc.initialize_state(model, geometry; rng = rng),
                            container, parameters; rng = rng)
        @test all(isfinite, container.data[:energy])
        @test all(isfinite, container.data[:acceptance_ratio])
    end

    # sweeps with different schemes write to different results directories
    @test ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry) !=
        ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(ucmc.SequentialProposal()),
                           geometry)
    @test ucmc.sweep_dirname(model, ucmc.HeatbathAlgorithm(), geometry) !=
        ucmc.sweep_dirname(model, ucmc.HeatbathAlgorithm(ucmc.SequentialProposal()),
                           geometry)
    @test ucmc.sweep_dirname(model, ucmc.OverrelaxationAlgorithm(), geometry) !=
        ucmc.sweep_dirname(model, ucmc.OverrelaxationAlgorithm(ucmc.SequentialProposal()),
                           geometry)
end

@testset "Checkerboard proposal" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    rng = Xoshiro(11)
    state = ucmc.initialize_state(model, geometry; rng = rng)

    checkerboard_algorithms = (ucmc.MetropolisAlgorithm(ucmc.CheckerboardProposal()),
                               ucmc.GlauberAlgorithm(ucmc.CheckerboardProposal()))

    # the coloring is the square-lattice parity mod(x + y, 2)
    colors = ucmc.checkerboard_colors(geometry)
    @test length(colors) == geometry.n_sites
    for site in 1:geometry.n_sites
        (loc, orbital) = lu.site_to_loc(site, geometry.unit_cell, geometry.lattice)
        @test colors[site] == mod(sum(loc), 2)
    end

    for algorithm in checkerboard_algorithms
        workspace = ucmc.algorithm_workspace(algorithm, model, geometry, state)
        @test workspace isa ucmc.LocalUpdateWorkspace
        @test workspace.proposal isa ucmc.ColoredProposalWorkspace

        # one sweep visits every site exactly once, color block by color block
        sites = [ucmc.propose(rng, algorithm, model, geometry, state; workspace = workspace).site
                 for _ in 1:geometry.n_sites]
        @test sort(sites) == collect(1:geometry.n_sites)
        half = geometry.n_sites ÷ 2
        @test all(==(0), colors[sites[1:half]])
        @test all(==(1), colors[sites[(half + 1):end]])
        # wraps around to the first color-0 site
        @test ucmc.propose(rng, algorithm, model, geometry, state; workspace = workspace).site ==
            findfirst(==(0), colors)

        # no bond connects two sites updated in the same half-sweep
        for block in (sites[1:half], sites[(half + 1):end])
            in_block = falses(geometry.n_sites)
            in_block[block] .= true
            for site in block
                @test !any(in_block[geometry.neighbor_table[site].neighbors])
            end
        end

        # a stateful scheme without its workspace is an explicit error
        @test_throws ArgumentError ucmc.propose(rng, algorithm, model, geometry, state)
        @test_throws ArgumentError ucmc.step!(algorithm, model, geometry, state, 2.0; rng = rng)

        # running totals stay exact through checkerboard sweeps
        cb_state = ucmc.initialize_state(model, geometry; rng = rng)
        cb_workspace = ucmc.algorithm_workspace(algorithm, model, geometry, cb_state)
        for _ in 1:(3 * geometry.n_sites)
            ucmc.step!(algorithm, model, geometry, cb_state, 2.0;
                       rng = rng, workspace = cb_workspace)
            @test cb_state.energy ≈ spin_energy(model, geometry, cb_state.spins)
            @test cb_state.magnetization == sum(cb_state.spins)
        end
    end

    # non-bipartite lattices are rejected eagerly, when the workspace is built
    triangular = triangular_geometry(3)
    tri_model = ucmc.IsingModel([1.0, 1.0, 1.0], [0.0])
    tri_state = ucmc.initialize_state(tri_model, triangular)
    for algorithm in checkerboard_algorithms
        @test_throws ArgumentError ucmc.algorithm_workspace(algorithm, tri_model, triangular, tri_state)
    end

    # the drivers build and thread the workspace automatically
    for algorithm in checkerboard_algorithms
        container = ucmc.MeasurementContainer(model, geometry, 4, 2;
                                              diagnostics = [:acceptance_ratio])
        parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
        ucmc.run_simulation(algorithm, model, geometry,
                            ucmc.initialize_state(model, geometry; rng = rng),
                            container, parameters; rng = rng)
        @test all(isfinite, container.data[:energy])
        @test all(isfinite, container.data[:acceptance_ratio])
    end

    # sweeps with different schemes write to different results directories
    @test ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(ucmc.CheckerboardProposal()),
                             geometry) !=
        ucmc.sweep_dirname(model, ucmc.MetropolisAlgorithm(), geometry)
end

@testset "Colored proposal" begin
    rng = Xoshiro(17)

    @test ucmc.ColoredProposal(3).ncolors == 3
    @test_throws ArgumentError ucmc.ColoredProposal(1)
    @test_throws ArgumentError ucmc.ColoredProposal(0)

    # tripartite coloring of the triangular lattice
    triangular = triangular_geometry(3)
    tri_model = ucmc.IsingModel([1.0, 1.0, 1.0], [0.0])
    tri_state = ucmc.initialize_state(tri_model, triangular; rng = rng)
    colors = ucmc.color_lattice(triangular, 3)
    @test sort(unique(colors)) == [1, 2, 3]
    for site in 1:triangular.n_sites
        for neighbor in triangular.neighbor_table[site].neighbors
            @test colors[site] != colors[neighbor]
        end
    end

    # the triangular lattice needs all three colors
    @test_throws ArgumentError ucmc.color_lattice(triangular, 2)

    colored_algorithms = (ucmc.MetropolisAlgorithm(ucmc.ColoredProposal(3)),
                          ucmc.GlauberAlgorithm(ucmc.ColoredProposal(3)))

    for algorithm in colored_algorithms
        workspace = ucmc.algorithm_workspace(algorithm, tri_model, triangular, tri_state)
        @test workspace isa ucmc.LocalUpdateWorkspace
        @test workspace.proposal isa ucmc.ColoredProposalWorkspace

        # one sweep visits every site exactly once, color block by color block
        sites = [ucmc.propose(rng, algorithm, tri_model, triangular, tri_state; workspace = workspace).site
                 for _ in 1:triangular.n_sites]
        @test sort(sites) == collect(1:triangular.n_sites)
        @test issorted(colors[sites])
        @test ucmc.propose(rng, algorithm, tri_model, triangular, tri_state; workspace = workspace).site ==
            findfirst(==(1), colors)

        # no bond connects two sites updated in the same color block
        for color in unique(colors)
            block = sites[colors[sites] .== color]
            in_block = falses(triangular.n_sites)
            in_block[block] .= true
            for site in block
                @test !any(in_block[triangular.neighbor_table[site].neighbors])
            end
        end

        # a stateful scheme without its workspace is an explicit error
        @test_throws ArgumentError ucmc.propose(rng, algorithm, tri_model, triangular, tri_state)
        @test_throws ArgumentError ucmc.step!(algorithm, tri_model, triangular, tri_state, 2.0; rng = rng)

        # running totals stay exact through colored sweeps
        col_state = ucmc.initialize_state(tri_model, triangular; rng = rng)
        col_workspace = ucmc.algorithm_workspace(algorithm, tri_model, triangular, col_state)
        for _ in 1:(3 * triangular.n_sites)
            ucmc.step!(algorithm, tri_model, triangular, col_state, 2.0;
                       rng = rng, workspace = col_workspace)
            @test col_state.energy ≈ spin_energy(tri_model, triangular, col_state.spins)
            @test col_state.magnetization == sum(col_state.spins)
        end
    end

    # a self-bond can never be colored
    square = square_geometry(3)
    self_table = deepcopy(square.neighbor_table)
    push!(self_table[1].bonds, -1)
    push!(self_table[1].templates, 1)
    push!(self_table[1].neighbors, 1)
    self_geometry = ucmc.Geometry(square.unit_cell, square.lattice, square.bonds,
                                  square.bond_id_to_template, square.bond_sources,
                                  self_table, square.n_sites)
    @test_throws ArgumentError ucmc.color_lattice(self_geometry, 3)

    # the drivers build and thread the workspace automatically
    for algorithm in colored_algorithms
        container = ucmc.MeasurementContainer(tri_model, triangular, 4, 2;
                                              diagnostics = [:acceptance_ratio])
        parameters = ucmc.SimulationParameters(2.0, 2, 4, 0, 2)
        ucmc.run_simulation(algorithm, tri_model, triangular,
                            ucmc.initialize_state(tri_model, triangular; rng = rng),
                            container, parameters; rng = rng)
        @test all(isfinite, container.data[:energy])
        @test all(isfinite, container.data[:acceptance_ratio])
    end

    # results directories distinguish colorings by count as well as name
    @test ucmc.sweep_dirname(tri_model, ucmc.MetropolisAlgorithm(ucmc.ColoredProposal(2)), triangular) !=
        ucmc.sweep_dirname(tri_model, ucmc.MetropolisAlgorithm(ucmc.ColoredProposal(3)), triangular)
    @test ucmc.sweep_dirname(tri_model, ucmc.MetropolisAlgorithm(ucmc.ColoredProposal(2)), triangular) !=
        ucmc.sweep_dirname(tri_model, ucmc.MetropolisAlgorithm(ucmc.CheckerboardProposal()), triangular)
end

@testset "Shared visiting order" begin
    geometry = square_geometry(4)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])

    # stateless schemes and non-local algorithms have no shareable data
    @test isnothing(ucmc.visiting_order(ucmc.RandomProposal(), geometry))
    @test isnothing(ucmc.visiting_order(ucmc.SequentialProposal(), geometry))
    @test isnothing(ucmc.shared_workspace(ucmc.MetropolisAlgorithm(), model, geometry))
    @test isnothing(ucmc.shared_workspace(ucmc.WolffAlgorithm(), model, geometry))

    for scheme in (ucmc.CheckerboardProposal(), ucmc.ColoredProposal(2))
        order = ucmc.visiting_order(scheme, geometry)
        @test length(order) == geometry.n_sites
        @test isperm(order)

        # a workspace built from the shared order carries that very vector,
        # and matches what a fresh coloring would produce
        workspace = ucmc.proposal_workspace(scheme, geometry, order)
        @test workspace isa ucmc.ColoredProposalWorkspace
        @test workspace.order === order
        @test ucmc.proposal_workspace(scheme, geometry).order == order

        # a bogus order is rejected: a non-permutation would silently skip or
        # revisit sites
        @test_throws ArgumentError ucmc.proposal_workspace(
            scheme, geometry, collect(1:(geometry.n_sites - 1)))
        @test_throws ArgumentError ucmc.proposal_workspace(
            scheme, geometry, fill(1, geometry.n_sites))

        # runs sharing one coloring match runs that re-color per simulation
        algorithm = ucmc.MetropolisAlgorithm(scheme)
        parameters = ucmc.SimulationParameters(2.0, 5, 20, 1, 4)
        shared = ucmc.shared_workspace(algorithm, model, geometry)
        @test shared == order
        datas = map((shared, nothing)) do sh
            rng = Xoshiro(2024)
            run_state = ucmc.initialize_state(model, geometry; rng = rng)
            container = ucmc.MeasurementContainer(model, geometry, 20, 4;
                                                  diagnostics = [:acceptance_ratio])
            ucmc.run_simulation(algorithm, model, geometry, run_state, container,
                                parameters; rng = rng, shared = sh)
            container.data
        end
        @test isequal(datas[1], datas[2])
    end

    # the coloring error of an uncolorable geometry is raised by
    # visiting_order — and by sweep_Ts directly, on the calling task
    triangular = triangular_geometry(3)
    tri_model = ucmc.IsingModel([1.0, 1.0, 1.0], [0.0])
    tri_state = ucmc.initialize_state(tri_model, triangular)
    @test_throws ArgumentError ucmc.visiting_order(ucmc.CheckerboardProposal(), triangular)
    @test_throws ArgumentError ucmc.sweep_Ts(
        ucmc.MetropolisAlgorithm(ucmc.CheckerboardProposal()), tri_model, triangular,
        tri_state, 2, 4, 0, 2, [2.0]; save = false)

    # a sweep with a colored proposal shares one coloring across its runs and
    # temperatures and stays reproducible
    algorithm = ucmc.MetropolisAlgorithm(ucmc.CheckerboardProposal())
    primaries(results) = [r.primary for r in results]
    seeded_sweep() = (Random.seed!(77); ucmc.sweep_Ts(
        algorithm, model, geometry,
        ucmc.initialize_state(model, geometry; rng = Xoshiro(5)),
        2, 10, 1, 2, [3.0, 2.0]; n_runs = 2, save = false, time_series = :none))
    @test isequal(primaries(seeded_sweep()), primaries(seeded_sweep()))
end

# Custom-algorithm/scheme fixtures for the workspace extension contracts:
# one with only the four-argument algorithm_workspace (the pre-sharing
# pattern), one with the documented defaulted-fifth-argument form, and a
# proposal scheme that exposes a visiting_order but no three-argument
# proposal_workspace.
struct FallbackWorkspaceAlg <: ucmc.AbstractAlgorithm end
struct FallbackWorkspace end
ucmc.algorithm_workspace(::FallbackWorkspaceAlg, model, geometry, state) = FallbackWorkspace()

struct DefaultedWorkspaceAlg <: ucmc.AbstractAlgorithm end
ucmc.shared_workspace(::DefaultedWorkspaceAlg, model, geometry) = 42
ucmc.algorithm_workspace(alg::DefaultedWorkspaceAlg, model, geometry, state,
                         shared = ucmc.shared_workspace(alg, model, geometry)) = shared

struct OrderExposingScheme <: ucmc.AbstractProposalScheme end
ucmc.proposal_workspace(::OrderExposingScheme, geometry) = :toy_workspace
ucmc.visiting_order(::OrderExposingScheme, geometry) = collect(1:geometry.n_sites)

@testset "Workspace extension contracts" begin
    geometry = square_geometry(2)
    model = ucmc.IsingModel([1.0, 1.0], [0.0])
    state = ucmc.initialize_state(model, geometry)

    # a custom algorithm with only a four-argument method keeps working
    # through the five-argument fallback: the shared data is discarded, the
    # workspace is not dropped, and nothing recurses
    @test ucmc.algorithm_workspace(FallbackWorkspaceAlg(), model, geometry, state) isa
        FallbackWorkspace
    @test ucmc.algorithm_workspace(FallbackWorkspaceAlg(), model, geometry, state, nothing) isa
        FallbackWorkspace

    # the documented defaulted-fifth-argument form receives the shared data
    @test ucmc.algorithm_workspace(DefaultedWorkspaceAlg(), model, geometry, state) == 42
    @test ucmc.algorithm_workspace(DefaultedWorkspaceAlg(), model, geometry, state, 7) == 7

    # a custom proposal scheme exposing visiting_order but no three-argument
    # proposal_workspace still builds a workspace (the order is discarded and
    # rebuilt per run); this used to be a MethodError
    alg = ucmc.MetropolisAlgorithm(OrderExposingScheme())
    @test ucmc.shared_workspace(alg, model, geometry) == collect(1:geometry.n_sites)
    workspace = ucmc.algorithm_workspace(alg, model, geometry, state)
    @test workspace isa ucmc.LocalUpdateWorkspace
    @test workspace.proposal == :toy_workspace
    @test workspace.acceptance isa ucmc.AcceptanceCache
    # and with the shared order passed explicitly
    @test ucmc.algorithm_workspace(alg, model, geometry, state,
                                   collect(1:geometry.n_sites)).proposal == :toy_workspace
end
