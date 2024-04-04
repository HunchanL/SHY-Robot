# SHY-Robot
<html>
<body>
<p><b>Codes for the SHY (Soft-Rigid-Hybrid) Robot Project</b></p>
<p><b>Sensor Calibration:</b>
  <ul>
    <li>Main.py: Initialize and runs the ionic resistive sensor calibration. Uncomment to calibrate a specific module. </li>
    <li>test_procedure.py: Includes functions that automatically runs the sensor calibration.</li>
  </ul>
<p><b>SHY continuum robot real time shape sensing:</b></p>
<ul>
  <li>Main.py: Runs the visualization of the continuum robot (roto-translational, bending, and translational module configuration)</li>
  <li>testProcedue.py: Includes necessary functions to initialize the test setup (EM tracker, syringe pump)</li>
  <li>Continuum.py: Includes necessary functions to inprepret the sensor calibration constants, computes homogeneous transformation matrices, and live plots the continnum robot shape</li>
  <li>pumplib.py: Includes commands to control Harvard Apparatus Syringe Pump (Pump 11 Elite)</li>
  <li>NDI.py: Includes commands to control Aurora Magnetic Tracker</li>
  <li>Demo:
    <li>PostProcess_image.py: Intreprets and plots the collected ionic resistive sensor dataset </li>
    <li>PostProcess_compute.py: Includes functions to read sensor dataset, and computes homogeneous tranformation matrices </li>
    <li>Consfiguration states.xlsx: Actual sample dataset presented in Figure 7 from the main manuscript </li>
  </li>
</ul>  
</body>
</html>
