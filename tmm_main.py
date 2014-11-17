
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import csv
import sys, os
import pylab
from scipy.integrate import simps

import tmm as tmm
import matplotlib.pyplot as plt

#define everything as a function of perovskite layer thickness and angle of incidence
def solar_results(activeLayer, AL_thickness,theta,atWL,calc_Efield,calc_Jsc,calc_Jsc_loss,calc_Q): #JB

	# list of layer thicknesses in nm
	d_list[activeLayer]=AL_thickness # JB
	angle=theta
	th0=angle*np.pi/180
	checkwavelength=atWL
	stepsize=1

	# allocate lists of y-values to plot
	R_a_s=[]
	R_a_p=[]
	T_a_s=[]
	T_a_p=[]
	A_a_s=[]
	A_a_p=[]
	E_s=[]
	E_p=[]
	E_xyz_tot=[]
	PA_s=[]
	PA_p=[]
	Gxl=[]
	Gxl_tot=[]
	Gxl_refl=[]
	Gxl_parasitic=[]

	for i,wl in enumerate(wavelengths):
		n_list=n_k[:,i]

		#calculate all data coherent (returns more values) and incoherent
		coh_tmm_data_s = tmm.coh_tmm('s',n_list, d_list, angle*np.pi/180, wl)
		coh_tmm_data_p = tmm.coh_tmm('p',n_list, d_list, angle*np.pi/180, wl)
		incoh_tmm_data_s = tmm.inc_tmm('s',n_list, d_list, c_list, angle*np.pi/180, wl)
		incoh_tmm_data_p = tmm.inc_tmm('p',n_list, d_list, c_list, angle*np.pi/180, wl)

		#use for R,T,A
		R_a_s.append(incoh_tmm_data_s['R'])
		R_a_p.append(incoh_tmm_data_p['R'])
		T_a_s.append(incoh_tmm_data_s['T'])
		T_a_p.append(incoh_tmm_data_p['T'])
		A_a_s.append(tmm.inc_absorp_in_each_layer(incoh_tmm_data_s))
		A_a_p.append(tmm.inc_absorp_in_each_layer(incoh_tmm_data_p))

		E_s.append([])
		E_p.append([])
		E_xyz_tot.append([])
		PA_s.append([])
		PA_p.append([])
		Gxl.append([])
		Gxl_tot.append([])
		Gxl_refl.append([])
		Gxl_parasitic.append([])

		#E-field for every layer different
		if calc_Efield==1:
			x_pos_abs=0
			t=[]
			for j in range(stackBegin,len(d_list)-1):
				if j==1:
					vw_s = incoh_tmm_data_s['VW_list'][j]
					kz_s = coh_tmm_data_s['kz_list'][j]
					vw_p = incoh_tmm_data_p['VW_list'][j]
					kz_p = coh_tmm_data_p['kz_list'][j]
				else:
					vw_s = coh_tmm_data_s['vw_list'][j]
					kz_s = coh_tmm_data_s['kz_list'][j]
					th_s = coh_tmm_data_s['th_list'][j]
					vw_p = coh_tmm_data_p['vw_list'][j]
					kz_p = coh_tmm_data_p['kz_list'][j]
					th_p = coh_tmm_data_p['th_list'][j]

				alpha=4*np.pi*np.imag(n_k[j,i])/wl


				#at every point x
				x_pos_rel=0
				for x in range(x_pos_abs,x_pos_abs+d_list[j]+1):
					t.append(x)

					E_plus_s=vw_s[0] * np.exp(1j * kz_s * x_pos_rel)
					E_minus_s=vw_s[1] * np.exp(-1j * kz_s * x_pos_rel)
					E_plus_p=vw_p[0] * np.exp(1j * kz_p * x_pos_rel)
					E_minus_p=vw_p[1] * np.exp(-1j * kz_p * x_pos_rel)

					E_pos_s=E_plus_s + E_minus_s
					E_pos_p=E_plus_p + E_minus_p

					E_s[i].append(E_pos_s)
					E_p[i].append(E_pos_p)

					E_z_pos=E_pos_s/np.sqrt(np.cos(th0))
					E_y_pos=E_pos_p*np.cos(th_p)/np.sqrt(np.cos(th0))
					E_x_pos=((-1)*E_plus_p + E_minus_p)*n_list[0]*np.sin(th0)/(n_list[j]*np.sqrt(np.cos(th0)))

					E_xyz_pos=0.5*np.square(np.absolute(E_z_pos))+0.5*(np.square(np.absolute(E_y_pos))+np.square(np.absolute(E_x_pos)))
					E_xyz_tot[i].append(E_xyz_pos)

					Q_pos_xyz=alpha*np.real(n_k[j,i])*power[i]*E_xyz_pos

					#calculate Energy dissipation and exciton generation rate
					if j==activeLayer:
						Gxl[i].append(Q_pos_xyz*1e-3*wl*1e-9/(h*c));
						Gxl_tot[i].append(Q_pos_xyz*1e-3*wl*1e-9/(h*c));

					else:
						Q_pos_xyz=alpha*np.real(n_k[j,i])*power[i]*(0.5*np.square(np.absolute(E_z_pos))+0.5*(np.square(np.absolute(E_y_pos))+np.square(np.absolute(E_x_pos))))
						Gxl_parasitic[i].append(Q_pos_xyz*1e-3*wl*1e-9/(h*c));
						Gxl_tot[i].append(Q_pos_xyz*1e-3*wl*1e-9/(h*c));

					R_frac=0.5*incoh_tmm_data_s['R']+0.5*incoh_tmm_data_p['R']
					Q_pos_xyz=Q_pos_xyz*(R_frac/(1-R_frac))
					Gxl_refl[i].append(Q_pos_xyz*1e-3*wl*1e-9/(h*c));

					#calculate pointing vectors
					#PA_s[i].append(tmm.position_resolved(j, x_pos_rel, coh_tmm_data_s)['poyn'])
					#PA_p[i].append(tmm.position_resolved(j, x_pos_rel, coh_tmm_data_p)['poyn'])
					x_pos_rel+=1

				x_pos_abs+=d_list[j]


	#convert all lists to arrays for plotting
	R_a_s=np.array(R_a_s)
	R_a_p=np.array(R_a_p)
	T_a_s=np.array(T_a_s)
	T_a_p=np.array(T_a_p)
	A_a_s=np.array(A_a_s)
	A_a_p=np.array(A_a_p)

	#get the 50% polarized fields
	R_a_tot=0.5*(R_a_s+R_a_p)
	T_a_tot=0.5*(T_a_s+T_a_p)
	A_tot=0.5*A_a_s+0.5*A_a_p
	A_tot2=A_tot # JB

	#plot A in each layer # JB
	plt.figure(6)
	plt.clf()
	plt.plot(wavelengths,A_tot2[:,activeLayer],'blue',label="A_perov")
	plt.xlabel('wavelength (nm)')
	plt.ylabel('A')
	plt.title('Our plot at %s'%angle)
	plt.legend()
	plt.show() 
	np.savetxt('A_tot.out', np.c_[wavelengths,A_tot], delimiter=',') # JB
    
	# delete first and last layer
	A_tot = np.delete(A_tot, 0, 1)  # delete first column of C
	A_tot = np.delete(A_tot, -1, 1)  # delete last column of C
	A_tot=A_tot.sum(axis=1)

    
	#save the data
	np.savetxt('RTA.out', np.c_[wavelengths,T_a_tot,R_a_tot,A_tot,T_a_tot+R_a_tot+A_tot], delimiter=',',header='Wavelength,T,R,A,Sum')
	
	#plot the results for R,T,A
	plt.close('all')
	plt.figure(1)
	plt.clf()
	plt.plot(wavelengths,T_a_tot,'blue',label="Transmission")
	plt.plot(wavelengths,R_a_tot,'purple',label="Reflection")
	plt.plot(wavelengths,A_tot,'red',label="Absorption")
	plt.plot(wavelengths,T_a_tot+R_a_tot+A_tot,'green',label="Sum")
	plt.xlabel('wl (nm)')
	plt.ylabel('Fraction reflected')
	plt.title('Our plot at %s'%angle)
	plt.legend()
	plt.ylim([0,0.2]) # JB
	plt.xlim([300,850]) # JB
	plt.show()


	if calc_Efield==1:
		t_a=np.array(t)
		i_faces= np.array(d_list)
		i_faces = np.delete(i_faces, 0)  # delete first column of C
		i_faces = np.delete(i_faces, 0)  # delete second column of C
		i_faces = np.delete(i_faces, -1)  # delete last column of C

		#calculate power of E-fields for s and p
		E_a_s=np.array(E_s)
		E_a_p=np.array(E_p)
		E_2_s=np.square(np.absolute(E_s[wavelengths.index(checkwavelength)]))
		E_2_p=np.square(np.absolute(E_p[wavelengths.index(checkwavelength)]))

		#save the data for the efield
		np.savetxt('E2_distribution.out', np.c_[t_a,E_2_s,E_2_p,E_2_s*0.5+E_2_p*0.5], delimiter=',',header='Position,E2_s,E2_p,E2_tot')

		#plot E field distrubtion
		plt.figure(2)
		plt.clf()
		plt.plot(t_a,E_2_s,'blue',label="s-pol")
		plt.plot(t_a,E_2_p,'red',label="p-pol")
		plt.plot(t_a,E_2_s*0.5+E_2_p*0.5,'green',label="Tot")
		plt.vlines(np.cumsum(i_faces), 0, 1, colors='k', linestyles='solid', label='')
		plt.xlabel('x (nm)')
		plt.ylabel('E2')
		plt.title('Our plot at %s'%angle)
		plt.legend()
		plt.show()


		#calculate generation and Jsc
		if calc_Jsc==1:
			lambdastep=1
			if np.size(wavelengths)>1:
				lambdastep=(np.amax(wavelengths)-np.amin(wavelengths))/(np.size(wavelengths)-1)

			Gx=np.sum(Gxl,axis=0)*lambdastep;
			Jsc=np.sum(Gx,axis=0)*stepsize*q*1e3
			print("Jsc:")
			print(Jsc)



			if calc_Jsc_loss==1:
				Gx_parasitic=np.sum(Gxl_parasitic,axis=0)*lambdastep;
				Jsc_parasitic=np.sum(Gx_parasitic,axis=0)*stepsize*q*1e3

				Gx_refl=np.sum(Gxl_refl,axis=0)*lambdastep;
				Jsc_refl=np.sum(Gx_refl,axis=0)*stepsize*q*1e3

				print("Jsc Loss from reflection:")
				print(Jsc_refl)
				print("Jsc loss from parasitic absorption:")
				print(Jsc_parasitic)

				#save the data for the generation
				np.savetxt('Jsc_data.out', np.c_[Jsc,Jsc_refl,Jsc_parasitic], delimiter=',',header='Jsc,Jsc_refl,Jsc_parasitic')

			if calc_Q==1:
				Gx_tot=np.sum(Gxl_tot,axis=0)*lambdastep;

				#save the data for the generation
				np.savetxt('G_distribution.out', np.c_[t_a,Gx_tot], delimiter=',',header='Position,G')

				plt.figure(3)
				plt.clf()
				plt.plot(t_a,Gx_tot,'blue')
				plt.vlines(np.cumsum(i_faces), 0, 1, colors='k', linestyles='solid', label='')
				plt.xlabel('x (nm)')
				plt.ylabel('Q')
				plt.title('Charge generation profile')
				plt.show()

			return Jsc
	return A_a_s, A_a_p, A_tot2 #JB 


def Jsc_dependence(angles,d_ALs):
	Jsc_results=[]

	for n,ang in enumerate(angles):
		Jsc_results.append([])
		for d_AL in d_ALs:
			print('Calc for angle and thickness:')
			print(ang)
			print(d_AL)
			Jsc_results[n].append(solar_results(d_AL,ang,550)[0]) #JB


	angles=np.array(angles)
	d_ALs=np.array(d_ALs)
	Jsc_results=np.array(Jsc_results)

	#save the data for the angular dependence
	np.savetxt('angular_dep.out', np.c_[angles,Jsc_results[:,0]], delimiter=',',header='angle,Jsc')

	#plot the final variation results
	plt.figure(4)
	plt.clf()
	plt.plot(angles,Jsc_results[:,0])
	plt.xlabel('theta (degrees)')
	plt.ylabel('Jsc (mA/cm^2)')
	plt.title('Different angles with 380nm thickness')
	plt.show()

	#save the data for the angular dependence
	np.savetxt('thickness_dep.out', np.c_[angles,Jsc_results[:,0]], delimiter=',',header='AL thickness,Jsc')
	plt.figure(5)
	plt.clf()
	plt.plot(d_ALs,Jsc_results[0,:])
	plt.xlabel('AL thickness (nm)')
	plt.ylabel('Jsc (mA/cm^2)')
	plt.title('Different AL thicknesses at 0 degrees')
	plt.show()

	return Jsc_results




# alternative for calculating Jsc thickness dep based on measured IQE #JB
def Jsc_t_dep(activeLayer,d_ALs,wavelengths,AM15_filename):
    Jsc_results2 = []
    nm = 1e-9
    
    IQE = np.genfromtxt('100EQE.txt',delimiter = '\t', dtype = float)
    power = import_AM15(AM15_filename)
    
    for d_AL in d_ALs:
        A_tot2 = solar_results(activeLayer,d_AL,0,550,0,0,0,0)[2]
        LHE = np.array(A_tot2[:,activeLayer])+np.array(A_tot2[:,activeLayer-1])
        sr = []
        ersr = []
        for i in range(len(LHE)):
            sr.append((LHE[i]*IQE[i,1])*wavelengths[i]*q*nm/(h*c*100))
            ersr.append(power[i]*10*sr[i])
        Jsc_results2.append(simps(ersr,wavelengths)*1000/10000)
        print(Jsc_results2)
    
    np.savetxt('thickness_dep.txt', np.c_[d_ALs,Jsc_results2], delimiter='\t')
    plt.figure(8)
    plt.clf()
    plt.plot(d_ALs,Jsc_results2)
    plt.xlabel('thickness (nm)')
    plt.ylabel('Jsc (mA/cm^2)')
    plt.title('Different thicknesses')
    plt.show()
    
    
    
# alternative for calculating Jsc angle dep based on measured IQE #JB
def Jsc_ang_dep(activeLayer,angles,d_list,wavelengths,AM15_filename):
    Jsc_results3_s = []
    Jsc_results3_p = []
    Jsc_results3 = []
    nm = 1e-9
    
    IQE = np.genfromtxt('IQE.txt',delimiter = '\t', dtype = float)
    power = import_AM15(AM15_filename)
    
    for angle in angles:
        rst = solar_results(activeLayer,d_list[activeLayer],angle,550,0,0,0,0)
        A_a_s = rst[0]
        A_a_p = rst[1]
        A_tot2 = rst[2]
        LHE_s = A_a_s[:,activeLayer]
        LHE_p = A_a_p[:,activeLayer]
        LHE = np.array(A_tot2[:,activeLayer])+np.array(A_tot2[:,activeLayer-1])
        sr_s = []
        ersr_s = []
        sr_p = []
        ersr_p = []
        sr = []
        ersr = []
        for i in range(len(LHE)):
            sr_s.append((LHE_s[i]*IQE[i,1])*wavelengths[i]*q*nm/(h*c*100))
            ersr_s.append(power[i]*10*sr_s[i])
            sr_p.append((LHE_p[i]*IQE[i,1])*wavelengths[i]*q*nm/(h*c*100))
            ersr_p.append(power[i]*10*sr_p[i])
            sr.append((LHE[i]*IQE[i,1])*wavelengths[i]*q*nm/(h*c*100))
            ersr.append(power[i]*10*sr[i])
        Jsc_results3_s.append(simps(ersr_s,wavelengths)*1000/10000)
        Jsc_results3_p.append(simps(ersr_p,wavelengths)*1000/10000)
        Jsc_results3.append(simps(ersr,wavelengths)*1000/10000)
        print(angle,Jsc_results3)
    
    np.savetxt('angle_dep_s.txt', np.c_[angles,Jsc_results3_s], delimiter='\t')
    np.savetxt('angle_dep_p.txt', np.c_[angles,Jsc_results3_p], delimiter='\t')
    np.savetxt('angle_dep.txt', np.c_[angles,Jsc_results3], delimiter='\t')
    plt.figure(9)
    plt.clf()
    plt.plot(angles,Jsc_results3_p)
    plt.xlabel('angle (deg)')
    plt.ylabel('Jsc (mA/cm^2)')
    plt.title('Different angles')
    plt.show()
    
    

#load the n/k data 
def import_refractive_indices(filename,layer_names):
	with open(filename) as f:
	    reader = csv.reader(f)
	    columns = next(reader)
	    colmap = dict(zip(columns, range(len(columns))))

	n_k_arr = np.loadtxt(filename, delimiter=",", skiprows=1)
	wavelengths_fromFile = n_k_arr[:,0].tolist()

	n_k_final=[]
	for layer_name in layers:
		row_n=np.interp(wavelengths, wavelengths_fromFile, n_k_arr[:, colmap[layer_name+'_n']])
		row_k=np.interp(wavelengths, wavelengths_fromFile, n_k_arr[:, colmap[layer_name+'_k']])
		n_k_final.append(row_n + 1j * row_k)

	n_k=np.array(n_k_final)
	return n_k

#load the AM15 data
def import_AM15(AM_filename):
	AM15_pre = np.loadtxt(AM_filename, delimiter=";", skiprows=1)


	wl_pre = AM15_pre[:,0]
	power_pre = AM15_pre[:,1]

	#interpolate the AM15 data to work with wl's from n_k_data
	power = np.interp(wavelengths, wl_pre, power_pre)
	return power


######################## user input ###########################

#define all constants and import data
#load values as array (for each wvlgenth), unpack = true means go for columns instead of rows
filename='n_k data.csv'
AM15_filename='AM15 spectrum.csv'
layers = ['Air','SodaLime','SnO2','SiO2','FTO','FTema','TPema','Perov','SpOxema','Au','Air']
d_list = [tmm.inf,2000,28,23,340,29,41,492,253,30,tmm.inf]
c_list = ['i','i','c','c','c','c','c','c','c','c','i']
#define active layer position (substrate is 0)
activeLayer=7
#define layer after which coherent stack begins (usually after substrate)
stackBegin=2

#input range for variation of angles
angles=range(0,90,5)
d_ALs=range(0,700,10)

#define constants
h=6.626e-34; #Js Planck's constant
c=2.998e8; #m/s speed of light
q=1.602e-19; #C electric charge
eps0=8.85418782e-12; #vacuum permitivity

#define wavelenght vector
wavelengths=np.arange(340,796).tolist()


n_k=import_refractive_indices(filename,layers)
power=import_AM15(AM15_filename)

#results_JSC=plot_dependence(angles,d_ALs)
jojo = solar_results(activeLayer,d_list[activeLayer],0,550,0,0,0,0)
#t_dep = Jsc_t_dep(activeLayer,d_ALs,wavelengths,AM15_filename)
#ang_dep = Jsc_ang_dep(activeLayer,angles,d_list,wavelengths,AM15_filename)




