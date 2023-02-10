import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('plotstyle.mplstyle')

#%% Set values

save_data = True
LH_on = True

# Spatial parameters
dr = 25                          # m, size of steps in r (horizontal)
dz = 25                          # m, size of steps in z (vertical)
dr2 = dr * dr
dz2 = dz * dz
window_height_m = 4000           # m, total vertical size of window
window_width_m = 12000           # m, total horizontal size of window

# Laccolith geometry
laccolith_a = 7500               # m, radius of the laccolith
laccolith_depth = 875            # m, overburden thickness on laccolith center

# Laccolith material parameters
P = 10.0**7.0                    # Pa, magma pressure driving laccolith
E = 25.0 * 10.0**9.0             # Pa, Young's modulus
nu = 0.25                        # dimensionless, Poisson's ratio

# Temperature parameters
T_hot = 1175.0                   # C, start hot temp; NOT MORE THAN LIQUIDUS
T_cool = -23.0                   # C, start cool temp
solidus = 1000                   # C, solidus of basalt
liquidus = 1200                  # C, liquidus of basalt

assert T_hot <= liquidus, "WARNING: T_hot must be set less than liquidus"

# Thermal material properties
conductivity = 1.5               # W/(m*K), thermal conductivity of basalt
rho = 3000                       # kg/m^3, density of basalt
c = 1000                         # J/(kg*K), specific heat capacity of basalt
alpha = conductivity / (rho * c) # m^2/s, thermal diffusivity of basalt

# Latent heat parameters
Q_per_m = 10.0**(5.0)                               # J/kg, latent heat for crystallizing basalt (Source: Lillis et al. 2009, JVGR)
total_LH_to_add = Q_per_m * (1.0 / c)               # K, total degrees of latent heat to add over the whole liquidus/solidus gap
LH_per_deg = total_LH_to_add / (liquidus - solidus) # K, degrees of latent heat to add per degree change in the temp array      

# Time parameters
run_time_yr = 50000                             # yr, total run time of simulation
sec_per_year = 31556952                         # s, seconds in a year           
run_time_sec = run_time_yr * sec_per_year       # s, total run time of simulation
dt_max = dr2 * dz2 / (2 * alpha * (dr2 + dz2))  # s, maximum stable time step for a given resolution
frac_of_dt = 1.0                                # dimensionless, fraction of maximum stable time step to use
dt = frac_of_dt * dt_max                        # s, time step to use for the rest of the script
nsteps = round(run_time_sec/dt)                 # dimensionless, total number of steps in simulation

#%% Create initial temperature array, laccolith-shaped intrusion

window_height_px = round(window_height_m / dz)
window_width_px = round(window_width_m / dr)

r_vals = np.linspace(dr, window_width_m, window_width_px)
col_indices = np.arange(0, window_width_px)

z_vals = np.linspace(dz, window_height_m, window_height_px)
row_indices = np.arange(0, window_height_px)

def laccolith_shape(x, a, P, E, h, nu):
    D = (E*(h**3.0))/(12.0*(1.0-(nu**2.0)))
    w = (P / (64.0 * D)) * (x**4.0 + a**4.0 - (2.0 * x**2.0 * a**2.0))
    return w

laccolith_max_w = laccolith_shape(0, laccolith_a, P, E, laccolith_depth, nu)

laccolith_a_px = round(laccolith_a / dr)
laccolith_depth_px = round(laccolith_depth / dz)
laccolith_max_w_px = round(laccolith_max_w / dz)
baseline = laccolith_depth + laccolith_max_w
baseline_px = round(baseline / dz)

initial_temps = T_cool * np.ones([window_height_px, window_width_px])
nrows = np.size(initial_temps, 0)
ncols = np.size(initial_temps, 1)

for z in range(0, nrows):
    for r in range(0, ncols):
        
        x = r * dr
        y = z * dz
        
        if y < laccolith_depth or y > baseline or x > laccolith_a:
            initial_temps[z,r] = T_cool
        else:
            w = laccolith_shape(x, laccolith_a, P, E, laccolith_depth, nu)
            boundary_z = baseline - w
            if y < boundary_z:
                initial_temps[z,r] = T_cool
            else:
                initial_temps[z,r] = T_hot
        
#%% Plot initial temperature array

ax = plt.subplot()
im = ax.imshow(initial_temps,
               extent = [0, r_vals[-1], z_vals[-1], 0],
               cmap = plt.get_cmap('hot'))
im.set_clim(T_cool, T_hot)
ax.ticklabel_format(useOffset = False)
plt.title('Initial temperature distribution')
plt.xlabel('Distance from axis (m)')
plt.ylabel('Depth (m)')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right',
                          size = '5%',
                          pad = 0.05)
cbar = plt.colorbar(im, cax = cax)
cbar.set_label('Temperature (C)')

if save_data:
    plt.savefig('Results/initial_temperatures.png', dpi = 300)
plt.show()

#%% Functions

# Heat propagation through time
def do_timestep(u0, T_ref, alpha, dr, dz, dt):
    
    u = np.ones_like(u0) * T_ref
    dr2 = dr * dr
    dz2 = dz * dz

    nrows = np.size(u0, 0)
    ncols = np.size(u0, 1)

    for z in range(1, nrows-1):
        for r in range(0, ncols-1):
            
            # First column of cells is the axis
            if r == 0:
                ur = 0.0
                urr = ((2.0 * u0[z,r+1]) - (2.0 * u0[z,r])) / dr2
                uzz = (u0[z-1,r] - (2.0 * u0[z,r]) + u0[z+1,r]) / dz2
                
                u[z,r] = u0[z,r] + (alpha * dt) * (urr + uzz)
            
            # For remaining columns of cells
            else:
                ur = (u0[z,r+1] - u0[z,r-1]) / (2.0 * dr)
                urr = (u0[z,r-1] - (2.0 * u0[z,r]) + u0[z,r+1]) / dr2
                uzz = (u0[z-1,r] - (2.0 * u0[z,r]) + u0[z+1,r]) / dz2
                
                axis_dist = r * dr
                u[z,r] = u0[z,r] + (alpha * dt) * \
                         (((1.0/axis_dist) * ur) + urr + uzz)
            
    return u

# Energy calculation over a whole array
def calc_energy(u, T_ref, dr, dz, rho, c):
    
    energy_array = np.zeros_like(u)
    dT = u - T_ref

    nrows = np.size(u, 0)
    ncols = np.size(u, 1)

    for z in range(0, nrows):
        for r in range(0, ncols):
            
            if r == 0:
                energy_array[z,r] = c * dT[z,r] * rho * (0.125 * dr * dr * dz)
                
            else:
                energy_array[z,r] = c * dT[z,r] * rho * (r * dr * dr * dz)
            
    energy_value = np.sum(energy_array)
            
    return energy_value

# Adding latent heat to an array
def add_latent_heat(u0, u,
                    liquidus, solidus,
                    LH_available, total_LH_to_add,
                    LH_per_deg, LH_added):
    
    u_with_LH = np.copy(u)
    
    nrows = np.size(u, 0)
    ncols = np.size(u, 1)
    
    for z in range(0, nrows-1):
        for r in range(0, ncols-1):
            
            u_new = u[z,r]
            u_old = u0[z,r]
            available = LH_available[z,r]
            
            if u_new < solidus and u_old < solidus:
                continue
            elif u_new > liquidus and u_old > liquidus:
                continue
            
            if u_new > u_old: # increase in temperature
                T_i = max(u_old, solidus)
                T_f = min(u_new, liquidus)
            else: # decrease or constant in temperature
                T_i = min(u_old, liquidus)
                T_f = max(u_new, solidus)
                
            delta_T = T_f - T_i
            LH_to_check = LH_per_deg * delta_T
            
            if delta_T > 0: # increase in temperature
                LH = min(total_LH_to_add - available, LH_to_check)
            else: # decrease or constant in temperature
                LH = max(-available, LH_to_check)
                
            u_with_LH[z,r] -= LH
            LH_available[z,r] += LH
            LH_added[z,r] -= LH
            
    return u_with_LH, LH_available
            
# Checking for new maximum temperatures
def check_for_max(u, max_temps):
    
    nrows = np.size(u, 0)
    ncols = np.size(u, 1)
    
    for z in range(0, nrows-1):
        for r in range(0, ncols-1):
            
            if u[z,r] > max_temps[z,r]:
                max_temps[z,r] = u[z,r]
                
    return max_temps

#%% Find the initial energy in the system

initial_energy_value = calc_energy(initial_temps, T_cool, dr, dz, rho, c)

system_energies = []
system_energies.append(initial_energy_value)

#%% Run the model

u0 = np.copy(initial_temps)
LH_available = np.maximum(np.copy(initial_temps) - solidus, 0) * LH_per_deg
LH_added = np.zeros_like(initial_temps)
max_temps = np.copy(initial_temps)

counter = 1
 
for i in range(nsteps):
    
    # Propagate the heat
    u = do_timestep(u0, T_cool, alpha, dr, dz, dt)
    
    # Add latent heat, if needed
    if LH_on:
        u, LH_available = add_latent_heat(u0, u,
                                          liquidus, solidus,
                                          LH_available, total_LH_to_add,
                                          LH_per_deg, LH_added)
        
    # Update the maximum temperatures array
    max_temps = check_for_max(u, max_temps)
    
    # Energy calculations
    energy = 0
    iteration_energy_value = calc_energy(u, T_cool, dr, dz, rho, c)
    system_energies.append(iteration_energy_value)
    
    u0 = u.copy()
    print('Completed step ', counter, '/', nsteps)
    counter += 1
    
final_temps = u.copy()

#%% Plot the temperature results

# Final temperatures
ax = plt.subplot()
im = ax.imshow(final_temps,
               extent = [0, r_vals[-1], z_vals[-1], 0],
               cmap = plt.get_cmap('hot'))
ax.ticklabel_format(useOffset = False)
plt.xlabel('Distance from axis (m)')
plt.ylabel('Depth (m)')
plt.title('Final temperatures')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right',
                          size = '5%',
                          pad = 0.05)
cbar = plt.colorbar(im, cax = cax)
cbar.set_label('Temperature (C)')

if save_data:
    plt.savefig('Results/final_temperatures_colors.png', dpi = 300)
plt.show()

# Maximum temperatures
ax = plt.subplot()
im = ax.imshow(max_temps,
               extent = [0, r_vals[-1], z_vals[-1], 0],
               cmap = plt.get_cmap('hot'))
im.set_clim(T_cool, T_hot)
ax.ticklabel_format(useOffset = False)
plt.xlabel('Distance from axis (m)')
plt.ylabel('Depth (m)')
plt.title('Maximum temperatures')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right',
                          size = '5%',
                          pad = 0.05)
cbar = plt.colorbar(im, cax = cax)
cbar.set_label('Temperature (C)')

if save_data:
    plt.savefig('Results/max_temperature.png', dpi = 300)
plt.show()

#%% Plot the energy results

# Calculate the percent change in energy
energy_x_axis = np.linspace(0, run_time_yr, len(system_energies))
energy_fractional_change = []
for i in range(len(system_energies)):
    base_val = np.abs(system_energies[0])
    first_val = system_energies[0]
    energy_fractional_change.append((system_energies[i] - first_val) / (base_val))
energy_fractional_change = np.array(energy_fractional_change)

# Plot the percent change in energy
plt.plot(energy_x_axis, energy_fractional_change)
plt.xlabel('Time (years)')
plt.ylabel('Energy (fractional change)')
plt.title('System energy')

if save_data:
    plt.savefig('Results/energy_tracking.png', dpi = 300)
plt.show()

#%% Plot the latent heat results

if LH_on:

    ax = plt.subplot(111)
    im = ax.imshow(LH_added,
                    extent = [0, r_vals[-1], z_vals[-1], 0],
                    cmap = plt.get_cmap('hot'))
    ax.ticklabel_format(useOffset=False)
    plt.xlabel('Degrees Longitude')
    plt.ylabel('Degrees Latitude')
    plt.title('Total added latent heat')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right',
                              size = '5%',
                              pad = 0.05)
    cbar = plt.colorbar(im, cax = cax)
    cbar.set_label('Added latent heat (K)')
    
    if save_data:
        plt.savefig('Results/LH_added.png', dpi = 600)
    plt.show()
    
#%% Save any remaining data if needed

if save_data:
    
    # Arrays
    np.save('Results/initial_temps.npy', initial_temps)
    np.save('Results/final_temps.npy', final_temps)
    np.save('Results/max_temps.npy', max_temps)
    
    np.save('Results/frac_energy.npy', energy_fractional_change)
    np.save('Results/frac_energy_times.npy', energy_x_axis)
    
    # Parameters
    data_file = open('Results/heat_flow_params.txt', 'w')
    data_file.write('dr = %f meters\n' % dr)
    data_file.write('dz = %f meters\n' % dz)
    data_file.write('window_height = %f meters\n' % window_height_m)
    data_file.write('window_width = %f meters\n' % window_width_m)
    data_file.write('laccolith_max_w = %f meters\n' % laccolith_max_w)
    data_file.write('laccolith_radius = %f meters\n' % laccolith_a)
    data_file.write('laccolith_depth = %f meters\n' % laccolith_depth)
    data_file.write('T_hot = %f C\n' % T_hot)
    data_file.write('T_cool = %f C\n' % T_cool)
    data_file.write('solidus = %f C\n' % solidus)
    data_file.write('liquidus = %f C\n' % liquidus)
    data_file.write('conductivity = %f W/(m*K)\n' % conductivity)
    data_file.write('rho = %f kg/m^3\n' % rho)
    data_file.write('c = %f J/(kg*K)\n' % c)
    data_file.write('alpha = %f m^2/s\n' % alpha)
    data_file.write('Q_per_m = %f J/kg\n' % Q_per_m)
    data_file.write('total_LH_to_add = %f K\n' % total_LH_to_add)
    data_file.write('run_time_yr = %f yr\n' % run_time_yr)
    data_file.write('dt = %f sec\n' % dt)
    data_file.write('nsteps = %f\n' % nsteps)
    data_file.close()
   