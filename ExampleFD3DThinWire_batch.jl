# ===========================================================================
# General Forward Solver GmshFEMInterface use
# ===========================================================================

using Revise

using GmshFEMInterface
using GmshFEMInterface: run_frequency_sweep
using GmshFEMInterface: default_callback_run_frequency_sweep_pre, default_callback_run_frequency_sweep_iter, default_callback_run_frequency_sweep_post
using GmshFEMInterface: default_callback_run_target_sweep_freqdomain_pre, default_callback_run_target_sweep_freqdomain_target_iter,
                        default_callback_run_target_sweep_freqdomain_freq_iter, default_callback_run_target_sweep_freqdomain_post
using GmshFEMInterface: create_sphere_data, create_no_target_data, create_layered_sphere_data

# ===========================================================================
# Main Program
# ===========================================================================

input_file = length(ARGS) >= 1 ? ARGS[1] : "./examples/ExampleFD3DThinWire/ExampleFD3DThinWire_batch.toml"
# ---------------------------------------------------------------------------
# Start by initializing the interface (initializes Gmsh)
# ---------------------------------------------------------------------------

GmshFEMInterface.initialize()

# ---------------------------------------------------------------------------
# Get solver options from TOML file
# ---------------------------------------------------------------------------

interface_options, geometry, physics, sources = GmshFEMInterface.get_input_options(input_file)

# ---------------------------------------------------------------------------
# Create solver
# ---------------------------------------------------------------------------
solver = GmshFEMInterface.create_solver(interface_options, geometry, physics, sources)

# ---------------------------------------------------------------------------
# Setup Output
# ---------------------------------------------------------------------------

output_handles = GmshFEMInterface.setup_output(interface_options, solver)

# ---------------------------------------------------------------------------
# Run  code
# ---------------------------------------------------------------------------

@time run_frequency_sweep(
    interface_options,
    solver,
    default_callback_run_frequency_sweep_pre,
    default_callback_run_frequency_sweep_iter,
    default_callback_run_frequency_sweep_post,
    output_handles)

GmshFEMInterface.finalize(output_handles)

