# lunar_heat_flow

The script 2D_axisymmetric_OWID.py (OWID = Our World in Data) is meant to be representative of my Python-writing skills. Broadly, the script models heat flow away from a volcanic intrusion. The following description of the code is an abbreviated version of the heat flow methods section in my manuscript titled "Volcanic thermal demagnetization of the Reiner Gamma magnetic anomaly", which is currently under review with the planetary science journal Icarus.

    The purpose of our thermal model of the laccolith is to determine which areas around the intrusion reach temperatures high enough (and for a long enough duration) to achieve any degree of demagnetization. To model the flow of heat away from the laccolith, we created a two-dimensional axisymmetric model where the axis of rotation runs vertically through the center of the model laccolith, the radial coordinate r runs perpendicularly outwards from this axis, and the vertical coordinate z runs parallel to this axis. The space is divided up into cells with radial resolution ∆r, vertical resolution ∆z, and temperature T. Heat flows through these cells according to the finite-difference version of the diffusion equation. The radial derivative of the temperatures of cells along the axis is zero (a Neumann boundary condition), and cells along the remaining three sides of our model space are held constant (a Dirichlet boundary condition).
    We define T_hot as the initial temperature of the intrusion in our model, and T_cool to be the temperature of the surrounding country rock before the heat from the intrusion begins to propagate. T_hot was chosen to be just below the liquidus of basalt, and T_cool was chosen to be the lunar mean near-surface temperature.
    The time step was set to be the maximum stable time step as determined by the spatial resolution of the model. For the values given in the script, the resulting maximum stable time step is 9.9 years, and the model was run to a time sufficient to reach the state where the laccolith has solidified and all cells in the domain reached their maximum temperatures (total run time was 50,000 years).
