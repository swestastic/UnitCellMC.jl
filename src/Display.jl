using Plots
using LatticeUtilities: UnitCell, Bond, Lattice, build_neighbor_table, map_neighbor_table, nsites, site_to_loc, loc_to_pos

function get_lattice_points(
    unit_cell::UnitCell,
    lattice::Lattice,
    )

    ns = nsites(unit_cell, lattice)
    positions = []
    orbitals = []
    for s in 1:ns
        # get the unit-cell location `l` and orbital `o` for site index s
        l, o = site_to_loc(s, unit_cell, lattice)
        # loc_to_pos returns the Cartesian position of orbital `o` at location `l`
        pos = loc_to_pos(l, o, unit_cell)
        push!(positions, pos)
        push!(orbitals, o)
    end
    # Convert positions to arrays for plotting
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]

    return xs, ys, orbitals, positions
end

function plot_lattice_points(
    xs,
    ys,
    orbitals
    )

    p = scatter(xs, ys, group=orbitals, markersize=6, legend=true, aspect_ratio=1, title="Lattice site positions with bonds")
    return p
end

function plot_lattice_bonds(
    p,
    neighbor_table_map,
    positions,
    unit_cell::UnitCell,
    lattice::Lattice
    )

    drawn_bonds = Set()
    ns = length(neighbor_table_map)

    # Build the neighbor table to get displacement information
    for site in 1:ns
        if haskey(neighbor_table_map, site)
            neighbors_data = neighbor_table_map[site]
            neighbors_list = neighbors_data.neighbors
            bonds_list = neighbors_data.bonds
            
            for (i, neighbor) in enumerate(neighbors_list)
                if neighbor > 0 && neighbor <= ns
                    bond_pair = tuple(sort([site, neighbor])...)
                    if bond_pair ∉ drawn_bonds
                        push!(drawn_bonds, bond_pair)
                        
                        # Get the bond information to check displacement
                        bond_id = bonds_list[i]
                        bond = neighbor_table_map[site].bonds[i]
                        
                        # Get displacement from site_to_loc
                        l1, o1 = site_to_loc(site, unit_cell, lattice)
                        l2, o2 = site_to_loc(neighbor, unit_cell, lattice)
                        
                        # Calculate displacement in lattice coordinates
                        disp = l2 - l1
                        
                        # Only draw bonds with small displacements (< half the lattice size in each direction)
                        # This filters out wraparound bonds
                        if all(abs(d) <= lattice.L[i]÷2 for (i,d) in enumerate(disp))
                            pos_site = positions[site]
                            pos_neighbor = positions[neighbor]
                            
                            # Draw the bond
                            x_vals = [pos_site[1], pos_neighbor[1]]
                            y_vals = [pos_site[2], pos_neighbor[2]]
                            plot!(p, x_vals, y_vals, color=:black, alpha=0.3, label="")
                        end
                    end
                end
            end
        end
    end

    return p
end


function plot_lattice_periodic_bonds(
    p,
    unit_cell::UnitCell,
    lattice::Lattice,
    neighbor_table_map,
    bonds
    )
    
    drawn_bonds = Set()

    # Get positions for the original lattice
    xs_orig, ys_orig, _, positions_orig = get_lattice_points(unit_cell, lattice)
    
    # Create new lattice with extended size
    lattice_new = Lattice(
        L = [lattice.L[1] + 1, lattice.L[2] + 1],
        periodic = [lattice.periodic[1], lattice.periodic[2]]
    )

    # Get positions for the new lattice
    xs_new, ys_new, _, positions_new = get_lattice_points(unit_cell, lattice_new)

    # Build neighbor table for the new lattice
    neighbor_table_new = build_neighbor_table(bonds, unit_cell, lattice_new)
    neighbor_table_map_new = map_neighbor_table(neighbor_table_new)

    # Create a mapping from position to original site index
    pos_to_orig_site = Dict()
    for s in 1:length(positions_orig)
        pos_to_orig_site[Tuple(positions_orig[s])] = s
    end
    
    # Calculate maximum bond length for local bonds
    min_distance = Inf
    for site in 1:length(positions_orig)
        for other_site in (site+1):length(positions_orig)
            if other_site <= length(positions_orig)
                dx = positions_orig[other_site][1] - positions_orig[site][1]
                dy = positions_orig[other_site][2] - positions_orig[site][2]
                dist = sqrt(dx^2 + dy^2)
                if dist > 0  # Exclude self-bonds
                    min_distance = min(min_distance, dist)
                end
            end
        end
    end
    max_bond_length = min_distance * 1.5
    
    # For each site in the NEW lattice that is at an original position
    for new_site in 1:length(positions_new)
        pos = Tuple(positions_new[new_site])
        
        # Check if this position corresponds to an original site
        if haskey(pos_to_orig_site, pos)
            orig_site = pos_to_orig_site[pos]
            
            if haskey(neighbor_table_map_new, new_site)
                neighbors_list = neighbor_table_map_new[new_site].neighbors
                
                for neighbor in neighbors_list
                    # Only draw if the neighbor is NOT at an original position
                    # (meaning it's in the extended region)
                    neighbor_pos = Tuple(positions_new[neighbor])
                    if !haskey(pos_to_orig_site, neighbor_pos)
                        bond_pair = tuple(sort([orig_site, neighbor])...)
                        if bond_pair ∉ drawn_bonds
                            push!(drawn_bonds, bond_pair)
                            
                            pos_site = positions_new[new_site]
                            pos_neighbor = positions_new[neighbor]
                            
                            # Calculate distance to verify it's a local bond (not wrapping)
                            dx = pos_neighbor[1] - pos_site[1]
                            dy = pos_neighbor[2] - pos_site[2]
                            dist = sqrt(dx^2 + dy^2)
                            
                            # Only draw if it's a local bond
                            if dist <= max_bond_length
                                # Draw the periodic bond with different styling
                                x_vals = [pos_site[1], pos_neighbor[1]]
                                y_vals = [pos_site[2], pos_neighbor[2]]
                                plot!(p, x_vals, y_vals, color=:red, alpha=0.6, label="", linewidth=2)
                            end
                        end
                    end
                end
            end
        end
    end
    
    return p
end

function plot_unit_cell_box(
    p,
    unit_cell::UnitCell
    )
    
    # Get the basis vectors (the three points in the unit cell)
    basis_vecs = unit_cell.basis_vecs
    
    # Extract x and y coordinates of the three basis points
    xs = [basis_vecs[i][1] for i in 1:length(basis_vecs)]
    ys = [basis_vecs[i][2] for i in 1:length(basis_vecs)]
    
    # Find bounding box: min and max coordinates
    x_min = minimum(xs)
    x_max = maximum(xs)
    y_min = minimum(ys)
    y_max = maximum(ys)
    
    # Add some padding to the box
    padding = 0.1
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    # Draw the bounding box with dashed lines
    # Bottom edge
    plot!(p, [x_min, x_max], [y_min, y_min], 
        color=:black, linestyle=:dash, alpha=0.5, label="", linewidth=1.5)
    
    # Right edge
    plot!(p, [x_max, x_max], [y_min, y_max], 
        color=:black, linestyle=:dash, alpha=0.5, label="", linewidth=1.5)
    
    # Top edge
    plot!(p, [x_max, x_min], [y_max, y_max], 
        color=:black, linestyle=:dash, alpha=0.5, label="", linewidth=1.5)
    
    # Left edge
    plot!(p, [x_min, x_min], [y_max, y_min], 
        color=:black, linestyle=:dash, alpha=0.5, label="", linewidth=1.5)
    
    return p
end

function plot_lattice_vectors(
    p,
    unit_cell::UnitCell
    )
    
    # Get the lattice vectors - they are stored as columns of a matrix
    lat_vecs = unit_cell.lattice_vecs
    lat_vec_1 = [lat_vecs[1, 1], lat_vecs[2, 1]]  # First column
    lat_vec_2 = [lat_vecs[1, 2], lat_vecs[2, 2]]  # Second column
    
    # Starting point for the arrows (origin)
    origin = [0.0, 0.0]
    
    # Draw arrow for first lattice vector
    quiver!(p, [origin[1]], [origin[2]], 
        quiver=([lat_vec_1[1]], [lat_vec_1[2]]),
        color=:blue, alpha=0.7, label="a₁", linewidth=2, arrow=:arrow)
    
    # Draw arrow for second lattice vector
    quiver!(p, [origin[1]], [origin[2]], 
        quiver=([lat_vec_2[1]], [lat_vec_2[2]]),
        color=:green, alpha=0.7, label="a₂", linewidth=2, arrow=:arrow)
    
    return p
end

function plot_lattice(
    unit_cell::UnitCell,
    lattice::Lattice,
    neighbor_table_map,
    bonds
    )

    xs, ys, orbitals, positions = get_lattice_points(unit_cell, lattice)

    # Draw lattice sites
    p = plot_lattice_points(xs, ys, orbitals)

    # Draw bonds between sites
    p = plot_lattice_bonds(p, neighbor_table_map, positions, unit_cell, lattice)
    
    # Draw periodic bonds
    p = plot_lattice_periodic_bonds(p, unit_cell, lattice, neighbor_table_map, bonds)
    
    # Draw unit cell box
    p = plot_unit_cell_box(p, unit_cell)
    
    # Draw lattice vectors
    p = plot_lattice_vectors(p, unit_cell)
    
    return p
end