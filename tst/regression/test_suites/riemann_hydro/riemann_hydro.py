# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2022, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import itertools
import utils.test_case
from scipy.optimize import curve_fit
from scipy import interpolate
from io import StringIO

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

method_cfgs = [
    {"nx1": 1024, "integrator": "vl2", "recon": "plm", "riemann": "hllc"},
    {"nx1": 64, "integrator": "vl2", "recon": "plm", "riemann": "hlle"},
    {"nx1": 64, "integrator": "vl2", "recon": "plm", "riemann": "hllc"},
    {"nx1": 64, "integrator": "rk3", "recon": "ppm", "riemann": "hlle"},
    {"nx1": 64, "integrator": "rk3", "recon": "ppm", "riemann": "hllc"},
]

# Following Toro Sec. 10.8 these are rho_l, u_l, p_l, rho_r, u_r, p_r, x0, and t_end, title
# Test are num 1, 6 and 7 from Table 10.1
# First test is a test where an analytic solution is available
init_cond_cfgs = [
    [
        1.0,
        0.0,
        1.0,
        0.125,
        0.0,
        0.1,
        1.0,
        0.245,
        "Sod with analytic solution",
    ],
    [
        1.0,
        0.75,
        1.0,
        0.125,
        0.0,
        0.1,
        1.0,
        0.2,
        "Sod with right shock,\nright contact,\nleft sonic rarefaction",
    ],
    [1.4, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, "Isolated stationary\ncontact"],
    [1.4, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 2.0, "Slow moving\nisolated contact"],
]

all_cfgs = list(itertools.product(method_cfgs, init_cond_cfgs))


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        method, init_cond = all_cfgs[step - 1]

        nx1 = method["nx1"]
        integrator = method["integrator"]
        recon = method["recon"]
        riemann = method["riemann"]

        rho_l = init_cond[0]
        u_l = init_cond[1]
        p_l = init_cond[2]
        rho_r = init_cond[3]
        u_r = init_cond[4]
        p_r = init_cond[5]
        x0 = init_cond[6]
        tlim = init_cond[7]

        # ensure that MeshBlock nx1 is <= 128 when using scratch (V100 limit on test system)
        mb_nx1 = nx1 // parameters.num_ranks
        while mb_nx1 > 128:
            mb_nx1 //= 2

        parameters.driver_cmd_line_args = [
            f"parthenon/mesh/nx1={nx1}",
            f"parthenon/meshblock/nx1={mb_nx1}",
            f"parthenon/time/integrator={integrator}",
            f"hydro/reconstruction={recon}",
            "parthenon/mesh/nghost=%d"
            % (3 if (recon == "ppm" or recon == "wenoz") else 2),
            f"hydro/riemann={riemann}",
            f"parthenon/output1/id={step}",
            f"problem/sod/rho_l={rho_l}",
            f"problem/sod/pres_l={p_l}",
            f"problem/sod/u_l={u_l}",
            f"problem/sod/rho_r={rho_r}",
            f"problem/sod/u_r={u_r}",
            f"problem/sod/pres_r={p_r}",
            f"problem/sod/x_discont={x0}",
            f"parthenon/time/tlim={tlim}",
            f"parthenon/output1/dt={tlim}",
        ]

        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        try:
            import phdf
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon hdf5 files.")
            return False

        test_success = True
        exact_rho = interpolate.interp1d(x_ex, rho_ex, fill_value=(1.0, 0.125), bounds_error=False)
        exact_v = interpolate.interp1d(x_ex, v_ex, fill_value=(0.0, 0.0), bounds_error=False)
        exact_p = interpolate.interp1d(x_ex, p_ex, fill_value=(0.1, 0.1), bounds_error=False)

        fig, p = plt.subplots(
            3, len(init_cond_cfgs), figsize=(3 * len(init_cond_cfgs), 8.0)
        )

        for step in range(len(all_cfgs)):
            method, init_cond = all_cfgs[step]
            col = init_cond_cfgs.index(init_cond)

            data_filename = f"{parameters.output_path}/parthenon.{step + 1}.final.phdf"
            data_file = phdf.phdf(data_filename)
            prim = data_file.Get("prim")
            rho = prim[0]
            vx = prim[1]
            pres = prim[4]
            zz, yy, xx = data_file.GetVolumeLocations()

            # Check against exact solution
            rho_check = exact_rho(xx)
            v_check = exact_v(xx)
            p_check = exact_p(xx)
            if step == 0:
              if (
                  any((rho_check - rho) > 0.05)
                  or any((v_check - vx) > 0.05)
                  or any((p_check - pres) > 0.05)
              ):
                  test_success = False
                  print(
                      f'{method["integrator"].upper()} {method["recon"].upper()} '
                      + f'{method["riemann"].upper()}' + ' failed.'
                  )

            label = (
                f'{method["integrator"].upper()} {method["recon"].upper()} '
                f'{method["riemann"].upper()}'
            )

            lw = 0.75
            p[0, col].plot(xx, rho, label=label, lw=lw)
            p[1, col].plot(xx, vx, label=label, lw=lw)
            p[2, col].plot(xx, pres, label=label, lw=lw)

        p[0, 0].plot(x_ex, rho_ex, label="exact", lw=0.75)
        p[1, 0].plot(x_ex, v_ex, label="exact", lw=0.75)
        p[2, 0].plot(x_ex, p_ex, label="exact", lw=0.75)

        p[0, 0].set_ylabel("rho")
        p[1, 0].set_ylabel("vx")
        p[2, 0].set_ylabel("press")

        for i in range(len(init_cond_cfgs)):
            p[-1, i].set_xlabel("x")
            p[0, i].set_title(init_cond_cfgs[i][-1])

        p[0, 0].legend(loc="upper left", bbox_to_anchor=(1.2 * len(init_cond_cfgs), 1))

        fig.savefig(
            os.path.join(parameters.output_path, "shock_tube.png"),
            bbox_inches="tight",
            dpi=300,
        )

        return test_success

# Exact solution to Sod right shock:
sod_solution = """
  -.1000000000000000E+01 0.1000000000000000E+01 0.0000000000000000E+00 0.1000000000000000E+01
  -.2920000000000000E+00 0.1000000000000000E+01 0.0000000000000000E+00 0.1000000000000000E+01
  -.2880000000000000E+00 0.9945846440720050E+00 0.6421460448575353E-02 0.9924267219159023E+00
  -.2840000000000000E+00 0.9831883841554278E+00 0.2002690262544608E-01 0.9765431422249107E+00
  -.2800000000000000E+00 0.9718968299745820E+00 0.3363234480231680E-01 0.9608779597816487E+00
  -.2760000000000000E+00 0.9607092583606245E+00 0.4723778697918753E-01 0.9454286663835448E+00
  -.2720000000000000E+00 0.9496249494822019E+00 0.6084322915605844E-01 0.9301927769257047E+00
  -.2680000000000000E+00 0.9386431868377304E+00 0.7444867133292918E-01 0.9151678292410118E+00
  -.2640000000000000E+00 0.9277632572476765E+00 0.8805411350979990E-01 0.9003513839409725E+00
  -.2600000000000000E+00 0.9169844508468379E+00 0.1016595556866706E+00 0.8857410242572966E+00
  -.2560000000000000E+00 0.9063060610766245E+00 0.1152649978635414E+00 0.8713343558842159E+00
  -.2520000000000000E+00 0.8957273846773387E+00 0.1288704400404121E+00 0.8571290068215360E+00
  -.2480000000000000E+00 0.8852477216804565E+00 0.1424758822172830E+00 0.8431226272184215E+00
  -.2440000000000000E+00 0.8748663754009082E+00 0.1560813243941537E+00 0.8293128892179117E+00
  -.2400000000000000E+00 0.8645826524293589E+00 0.1696867665710244E+00 0.8156974868021667E+00
  -.2360000000000000E+00 0.8543958626244899E+00 0.1832922087478952E+00 0.8022741356384400E+00
  -.2320000000000000E+00 0.8443053191052786E+00 0.1968976509247660E+00 0.7890405729257775E+00
  -.2280000000000000E+00 0.8343103382432802E+00 0.2105030931016367E+00 0.7759945572424408E+00
  -.2240000000000000E+00 0.8244102396549075E+00 0.2241085352785074E+00 0.7631338683940532E+00
  -.2200000000000000E+00 0.8146043461937122E+00 0.2377139774553783E+00 0.7504563072624655E+00
  -.2160000000000000E+00 0.8048919839426661E+00 0.2513194196322490E+00 0.7379596956553420E+00
  -.2120000000000000E+00 0.7952724822064409E+00 0.2649248618091198E+00 0.7256418761564637E+00
  -.2080000000000000E+00 0.7857451735036893E+00 0.2785303039859905E+00 0.7135007119767458E+00
  -.2040000000000000E+00 0.7763093935593263E+00 0.2921357461628613E+00 0.7015340868059714E+00
  -.2000000000000000E+00 0.7669644812968097E+00 0.3057411883397321E+00 0.6897399046652356E+00
  -.1960000000000000E+00 0.7577097788304196E+00 0.3193466305166028E+00 0.6781160897600992E+00
  -.1919999999999999E+00 0.7485446314575414E+00 0.3329520726934735E+00 0.6666605863344556E+00
  -.1879999999999999E+00 0.7394683876509449E+00 0.3465575148703444E+00 0.6553713585250996E+00
  -.1840000000000001E+00 0.7304803990510659E+00 0.3601629570472147E+00 0.6442463902170045E+00
  -.1800000000000000E+00 0.7215800204582865E+00 0.3737683992240854E+00 0.6332836848993014E+00
  -.1760000000000000E+00 0.7127666098252160E+00 0.3873738414009563E+00 0.6224812655219623E+00
  -.1720000000000000E+00 0.7040395282489718E+00 0.4009792835778270E+00 0.6118371743531806E+00
  -.1680000000000000E+00 0.6953981399634597E+00 0.4145847257546978E+00 0.6013494728374532E+00
  -.1640000000000000E+00 0.6868418123316554E+00 0.4281901679315686E+00 0.5910162414543568E+00
  -.1600000000000000E+00 0.6783699158378851E+00 0.4417956101084393E+00 0.5808355795780212E+00
  -.1560000000000000E+00 0.6699818240801048E+00 0.4554010522853101E+00 0.5708056053372933E+00
  -.1520000000000000E+00 0.6616769137621835E+00 0.4690064944621808E+00 0.5609244554765968E+00
  -.1480000000000000E+00 0.6534545646861823E+00 0.4826119366390516E+00 0.5511902852174778E+00
  -.1440000000000000E+00 0.6453141597446359E+00 0.4962173788159224E+00 0.5416012681208416E+00
  -.1400000000000000E+00 0.6372550849128327E+00 0.5098228209927931E+00 0.5321555959498732E+00
  -.1360000000000000E+00 0.6292767292410960E+00 0.5234282631696640E+00 0.5228514785336451E+00
  -.1320000000000000E+00 0.6213784848470649E+00 0.5370337053465346E+00 0.5136871436314069E+00
  -.1280000000000000E+00 0.6135597469079747E+00 0.5506391475234054E+00 0.5046608367975564E+00
  -.1240000000000000E+00 0.6058199136529379E+00 0.5642445897002761E+00 0.4957708212472899E+00
  -.1200000000000000E+00 0.5981583863552248E+00 0.5778500318771469E+00 0.4870153777229320E+00
  -.1160000000000000E+00 0.5905745693245443E+00 0.5914554740540178E+00 0.4783928043609393E+00
  -.1120000000000000E+00 0.5830678698993250E+00 0.6050609162308884E+00 0.4699014165595806E+00
  -.1080000000000000E+00 0.5756376984389954E+00 0.6186663584077592E+00 0.4615395468472886E+00
  -.1040000000000000E+00 0.5682834683162649E+00 0.6322718005846300E+00 0.4533055447516833E+00
  -.9999999999999998E-01 0.5610045959094045E+00 0.6458772427615007E+00 0.4451977766692644E+00
  -.9599999999999997E-01 0.5538005005945283E+00 0.6594826849383715E+00 0.4372146257357714E+00
  -.9199999999999997E-01 0.5466706047378729E+00 0.6730881271152422E+00 0.4293544916972101E+00
  -.8799999999999997E-01 0.5396143336880790E+00 0.6866935692921130E+00 0.4216157907815423E+00
  -.8399999999999996E-01 0.5326311157684727E+00 0.7002990114689838E+00 0.4139969555710391E+00
  -.7999999999999996E-01 0.5257203822693443E+00 0.7139044536458544E+00 0.4064964348752932E+00
  -.7599999999999996E-01 0.5188815674402314E+00 0.7275098958227253E+00 0.3991126936048933E+00
  -.7199999999999995E-01 0.5121141084821985E+00 0.7411153379995961E+00 0.3918442126457528E+00
  -.6799999999999995E-01 0.5054174455401176E+00 0.7547207801764668E+00 0.3846894887340953E+00
  -.6399999999999995E-01 0.4987910216949495E+00 0.7683262223533376E+00 0.3776470343320946E+00
  -.6000000000000005E-01 0.4922342829560242E+00 0.7819316645302079E+00 0.3707153775041658E+00
  -.5600000000000005E-01 0.4857466782533217E+00 0.7955371067070787E+00 0.3638930617939076E+00
  -.5200000000000005E-01 0.4793276594297530E+00 0.8091425488839494E+00 0.3571786461016933E+00
  -.4800000000000004E-01 0.4729766812334404E+00 0.8227479910608203E+00 0.3505707045629084E+00
  -.4400000000000004E-01 0.4666932013099990E+00 0.8363534332376910E+00 0.3440678264268340E+00
  -.4000000000000004E-01 0.4604766801948169E+00 0.8499588754145617E+00 0.3376686159361740E+00
  -.3600000000000003E-01 0.4543265813053354E+00 0.8635643175914325E+00 0.3313716922072227E+00
  -.3200000000000003E-01 0.4482423709333314E+00 0.8771697597683034E+00 0.3251756891106752E+00
  -.2800000000000002E-01 0.4422235182371966E+00 0.8907752019451741E+00 0.3190792551530741E+00
  -.2400000000000002E-01 0.4362694952342192E+00 0.9043806441220448E+00 0.3130810533588938E+00
  -.2000000000000002E-01 0.4303797767928642E+00 0.9179860862989155E+00 0.3071797611532595E+00
  -.1600000000000001E-01 0.4263194281784952E+00 0.9274526200489498E+00 0.3031301780506468E+00
  0.2240000000000000E+00 0.4263194281784952E+00 0.9274526200489498E+00 0.3031301780506468E+00
  0.2280000000000000E+00 0.2655737117053071E+00 0.9274526200489498E+00 0.3031301780506468E+00
  0.4279999999999999E+00 0.2655737117053071E+00 0.9274526200489498E+00 0.3031301780506468E+00
  0.4319999999999999E+00 0.1250000000000000E+00 0.0000000000000000E+00 0.1000000000000000E+00
  0.9960000000000000E+00 0.1250000000000000E+00 0.0000000000000000E+00 0.1000000000000000E+00
  """

x_ex, rho_ex, v_ex, p_ex = np.genfromtxt(StringIO(sod_solution)).T
x_ex = (x_ex + 1.0)