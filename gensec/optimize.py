"""Structure optimization. 
"""
import time
import numpy as np
from gensec.precon import preconditioned_hessian, Kabsh_rmsd
from ase.optimize.bfgs import BFGS
from numpy import eye, sqrt, isinf
from ase.optimize import BFGSLineSearch
from ase.optimize import LBFGS
from ase.optimize.precon import PreconLBFGS


class PreconLBFGS_mod(PreconLBFGS):

    """Summary

    Attributes:
        f0 (TYPE): Description
        fixed_frame (TYPE): Description
        Hinv (TYPE): Description
        initial (TYPE): Description
        molindixes (TYPE): Description
        p (TYPE): Description
        parameters (TYPE): Description
        r0 (TYPE): Description
        rmsd_dev (TYPE): Description
        structure (TYPE): Description
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=None,
        memory=100,
        damping=1.0,
        alpha=70.0,
        master=None,
        precon="ID",
        variable_cell=False,
        use_armijo=False,
        c1=0.23,
        c2=0.46,
        a_min=None,
        rigid_units=None,
        rotation_factors=None,
        Hinv=None,
        structure=None,
        H0=None,
        fixed_frame=None,
        parameters=None,
        initial=None,
        rmsd_dev=1000.0,
        molindixes=None,
    ):
        """Summary

        Args:
            atoms (TYPE): Description
            restart (None, optional): Description
            logfile (str, optional): Description
            trajectory (None, optional): Description
            maxstep (None, optional): Description
            memory (int, optional): Description
            damping (float, optional): Description
            alpha (float, optional): Description
            master (None, optional): Description
            precon (str, optional): Description
            variable_cell (bool, optional): Description
            use_armijo (bool, optional): Description
            c1 (float, optional): Description
            c2 (float, optional): Description
            a_min (None, optional): Description
            rigid_units (None, optional): Description
            rotation_factors (None, optional): Description
            Hinv (None, optional): Description
            structure (None, optional): Description
            H0 (None, optional): Description
            fixed_frame (None, optional): Description
            parameters (None, optional): Description
            initial (None, optional): Description
            rmsd_dev (float, optional): Description
            molindixes (None, optional): Description
        """
        PreconLBFGS.__init__(
            self,
            atoms,
            use_armijo=use_armijo,
            precon=precon,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=None,
        )

        self.Hinv = np.linalg.inv(H0)  # initial hessian
        # self.Hinv = None
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters
        # self.precon = Exp(mu=self.structure.mu, A=3.0, recalc_mu=False)
        self.initial = initial
        self.rmsd_dev = rmsd_dev
        self.molindixes = molindixes

    def step(self, f=None):
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it

        Args:
            f (None, optional): Description
        """
        r = self.atoms.get_positions()

        if f is None:
            f = self.atoms.get_forces()

        print(self._just_reset_hessian)
        previously_reset_hessian = self._just_reset_hessian
        self.update(r, f, self.r0, self.f0)

        s = self.s
        y = self.y
        rho = self.rho
        H0 = self.H0

        loopmax = np.min([self.memory, len(self.y)])
        a = np.empty((loopmax,), dtype=np.float64)

        # The algorithm itself:
        q = -f.reshape(-1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = rho[i] * np.dot(s[i], q)
            q -= a[i] * y[i]

        # if self.initial: # Update the Hessian (not inverse!!!)
        # print("Energy", self.atoms.get_potential_energy(), "Force   ",  fmax)
        # Calculate RMSD between current and initial steps:
        if (
            Kabsh_rmsd(self.atoms, self.initial, self.molindixes)
            > self.rmsd_dev
            and self.parameters["calculator"]["preconditioner"]["rmsd_update"][
                "activate"
            ]
        ):
            print("################################Applying update")
            self.Hinv = np.linalg.inv(
                preconditioned_hessian(
                    self.structure,
                    self.fixed_frame,
                    self.parameters,
                    self.atoms,
                    np.eye(3 * len(self.atoms)),
                    task="initial",
                )
            )

            z = np.dot(self.Hinv, q)
            a0 = self.atoms.copy()
            self.initial = a0
            self.reset_hessian()
        else:
            z = np.dot(self.Hinv, q)

        for i in range(loopmax):
            b = rho[i] * np.dot(y[i], z)
            z += s[i] * (a[i] - b)

        self.p = -z.reshape((-1, 3))
        ###

        g = -f
        if self.e1 is not None:
            e = self.e1
        else:
            e = self.func(r)
        self.line_search(r, g, e, previously_reset_hessian)
        dr = (self.alpha_k * self.p).reshape(len(self.atoms), -1)

        if self.alpha_k != 0.0:
            self.atoms.set_positions(r + dr)

        self.iteration += 1
        self.r0 = r
        self.f0 = -g
        self.dump(
            (
                self.iteration,
                self.s,
                self.y,
                self.rho,
                self.r0,
                self.f0,
                self.e0,
                self.task,
            )
        )


class BFGS_mod(BFGS):

    """Summary

    Attributes:
        fixed_frame (TYPE): Description
        H (TYPE): Description
        H0 (TYPE): Description
        initial (TYPE): Description
        molindixes (TYPE): Description
        parameters (TYPE): Description
        rmsd_dev (TYPE): Description
        structure (TYPE): Description
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=0.04,
        master=None,
        initial=None,
        rmsd_dev=1000.0,
        molindixes=None,
        structure=None,
        H0=None,
        fixed_frame=None,
        parameters=None,
        mu=None,
        A=None,
        known=None,
    ):
        """Summary

        Args:
            atoms (TYPE): Description
            restart (None, optional): Description
            logfile (str, optional): Description
            trajectory (None, optional): Description
            maxstep (float, optional): Description
            master (None, optional): Description
            initial (None, optional): Description
            rmsd_dev (float, optional): Description
            molindixes (None, optional): Description
            structure (None, optional): Description
            H0 (None, optional): Description
            fixed_frame (None, optional): Description
            parameters (None, optional): Description
            mu (None, optional): Description
            A (None, optional): Description
            known (None, optional): Description
        """
        BFGS.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=None,
        )

        self.H0 = H0  # initial hessian
        self.initial = initial
        self.rmsd_dev = rmsd_dev
        self.molindixes = molindixes
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters

    def update(self, r, f, r0, f0):
        """Summary

        Args:
            r (TYPE): Description
            f (TYPE): Description
            r0 (TYPE): Description
            f0 (TYPE): Description

        Returns:
            TYPE: Description
        """
        if self.H is None:
            self.H = self.H0  # This is Heeian - not inverse!!!
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b

        if self.initial:  # Update the Hessian (not inverse!!!)
            # print("Energy", self.atoms.get_potential_energy(), "Force   ",  fmax)
            # Calculate RMSD between current and initial steps:
            if (
                Kabsh_rmsd(self.atoms, self.initial, self.molindixes)
                > self.rmsd_dev
            ):
                print("################################Applying update")
                self.H = preconditioned_hessian(
                    self.structure,
                    self.fixed_frame,
                    self.parameters,
                    self.atoms,
                    self.H,
                    task="update",
                )

                a0 = self.atoms.copy()
                self.initial = a0


class BFGSLineSearch_mod(BFGSLineSearch):

    """Summary

    Attributes:
        fixed_frame (TYPE): Description
        fmax_last (TYPE): Description
        H (TYPE): Description
        H0 (TYPE): Description
        I (TYPE): Description
        initial (TYPE): Description
        molindixes (TYPE): Description
        parameters (TYPE): Description
        rmsd_dev (TYPE): Description
        steps_in_row (int): Description
        structure (TYPE): Description
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=None,
        master=None,
        initial=None,
        rmsd_dev=1000.0,
        molindixes=None,
        structure=None,
        H0=None,
        fixed_frame=None,
        parameters=None,
        mu=None,
        A=None,
        known=None,
        c1=0.23,
        c2=0.46,
        alpha=10.0,
        stpmax=50.0,
        force_consistent=True,
    ):
        """Summary

        Args:
            atoms (TYPE): Description
            restart (None, optional): Description
            logfile (str, optional): Description
            trajectory (None, optional): Description
            maxstep (None, optional): Description
            master (None, optional): Description
            initial (None, optional): Description
            rmsd_dev (float, optional): Description
            molindixes (None, optional): Description
            structure (None, optional): Description
            H0 (None, optional): Description
            fixed_frame (None, optional): Description
            parameters (None, optional): Description
            mu (None, optional): Description
            A (None, optional): Description
            known (None, optional): Description
            c1 (float, optional): Description
            c2 (float, optional): Description
            alpha (float, optional): Description
            stpmax (float, optional): Description
            force_consistent (bool, optional): Description
        """
        BFGSLineSearch.__init__(
            self,
            atoms,
            restart=None,
            logfile=logfile,
            maxstep=maxstep,
            trajectory=trajectory,
            c1=c1,
            c2=c2,
            alpha=alpha,
            stpmax=stpmax,
            master=None,
            force_consistent=force_consistent,
        )

        self.H0 = H0
        self.initial = initial
        self.rmsd_dev = rmsd_dev
        self.molindixes = molindixes
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters
        self.steps_in_row = 0
        self.fmax_last = None

    def update(self, r, g, r0, g0, p0):
        """Summary

        Args:
            r (TYPE): Description
            g (TYPE): Description
            r0 (TYPE): Description
            g0 (TYPE): Description
            p0 (TYPE): Description

        Returns:
            TYPE: Description
        """
        self.I = eye(len(self.atoms) * 3, dtype=int)
        if self.H is None:
            self.H = np.linalg.inv(self.H0)  # This is inverse Hessian!!!
            # if np.array_equal(self.H0, np.eye(3 * len(self.atoms)) * 70):
            # Like in default ASE
            # self.H = eye(3 * len(self.atoms))/70.
            # self.B = np.linalg.inv(self.H)
            return
        else:
            if self.fmax_last is None:
                self.fmax_last = sqrt(
                    (g0.reshape(-1, 3) ** 2).sum(axis=1).max()
                )

            dr = r - r0
            dg = g - g0
            # self.alpha_k can be None!!!
            if not (
                (
                    (self.alpha_k or 0) > 0
                    and abs(np.dot(g, p0)) - abs(np.dot(g0, p0)) < 0
                )
                or self.replay
            ):
                return
            if self.no_update is True:
                print("skip update")
                return

            try:  # this was handled in numeric, let it remain for more safety
                rhok = 1.0 / (np.dot(dg, dr))
            except ZeroDivisionError:
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            if isinf(rhok):  # this is patch for np
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            A1 = self.I - dr[:, np.newaxis] * dg[np.newaxis, :] * rhok
            A2 = self.I - dg[:, np.newaxis] * dr[np.newaxis, :] * rhok
            self.H = (
                np.dot(A1, np.dot(self.H, A2))
                + rhok * dr[:, np.newaxis] * dr[np.newaxis, :]
            )

            # # self.B = np.linalg.inv(self.H)
            if self.parameters["calculator"]["preconditioner"]["rmsd_update"][
                "activate"
            ]:
                # self.steps_in_raw+=1
                # or self.steps_in_raw > 50
                current_fmax = sqrt((g.reshape(-1, 3) ** 2).sum(axis=1).max())
                rmsd = Kabsh_rmsd(self.atoms, self.initial, self.molindixes)
                # if current_fmax > self.fmax_last:
                # self.fmax_last = current_fmax
                if self.fmax_last / current_fmax > 3 or rmsd > self.rmsd_dev:
                    # # Calculate RMSD between current and initial steps:
                    # if Kabsh_rmsd(self.atoms, self.initial, self.molindixes) > self.rmsd_dev:
                    print(
                        "################################Applying update",
                        current_fmax,
                        self.steps_in_row,
                        rmsd,
                    )
                    self.fmax_last = current_fmax

                    self.H = np.linalg.inv(
                        preconditioned_hessian(
                            self.structure,
                            self.fixed_frame,
                            self.parameters,
                            self.atoms,
                            self.H,
                            task="update",
                        )
                    )

                    #         fig, ax = plt.subplots()
                    #         im = ax.imshow(self.H)
                    #         plt.colorbar(im)
                    #         np.savetxt(f+"Exp_after.hes", self.H)
                    #         # ax.set_title("Difference in the Hessians")
                    #         # fig.tight_layout()
                    #         # plt.savefig("diff_approx.png", dpi=300)
                    #         plt.show()
                    #         plt.clf()

                    #         # self.H = ASR(self.H)
                    #         # # Add stabilization to the diagonal
                    #         # # jitter = 0.005
                    #         # # self.H = add_jitter(self.H, jitter)
                    #         # # Check if positive and symmetric:
                    #         # symmetric, positive = check_positive_symmetric(self.H)
                    #         # if not symmetric:
                    #         #     print("Hessian is not symmetric! Will give troubles during optimization!")
                    #         #     # sys.exit(0)
                    #         # if not positive:
                    #         #     print("Hessian is not positive definite! Will give troubles during optimization!")
                    a0 = self.atoms.copy()
                    self.initial = a0

            #         # fig, ax = plt.subplots()
            #         # im = ax.imshow(self.H)
            #         # plt.colorbar(im)

            #         # ax.set_title("Difference in the Hessians")
            #         # fig.tight_layout()
            #         # plt.savefig("diff_approx.png", dpi=300)
            #         # plt.show()
            #         # plt.clf()

            # symmetric, positive = check_positive_symmetric(self.H)
            # print(symmetric, positive)


class LBFGS_Linesearch_mod(LBFGS):

    """Summary

    Attributes:
        f0 (TYPE): Description
        fixed_frame (TYPE): Description
        fmax_last (TYPE): Description
        Hinv (TYPE): Description
        initial (TYPE): Description
        iteration (int): Description
        molindixes (TYPE): Description
        p (TYPE): Description
        parameters (TYPE): Description
        r0 (TYPE): Description
        rho (list): Description
        rmsd_dev (TYPE): Description
        s (list): Description
        steps_in_row (int): Description
        structure (TYPE): Description
        y (list): Description
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=None,
        master=None,
        initial=None,
        rmsd_dev=1000.0,
        molindixes=None,
        structure=None,
        H0_init=None,
        fixed_frame=None,
        parameters=None,
        mu=None,
        A=None,
        known=None,
        alpha=70.0,
        damping=1.0,
        memory=100,
        force_consistent=True,
        use_line_search=True,
    ):
        """Summary

        Args:
            atoms (TYPE): Description
            restart (None, optional): Description
            logfile (str, optional): Description
            trajectory (None, optional): Description
            maxstep (None, optional): Description
            master (None, optional): Description
            initial (None, optional): Description
            rmsd_dev (float, optional): Description
            molindixes (None, optional): Description
            structure (None, optional): Description
            H0_init (None, optional): Description
            fixed_frame (None, optional): Description
            parameters (None, optional): Description
            mu (None, optional): Description
            A (None, optional): Description
            known (None, optional): Description
            alpha (float, optional): Description
            damping (float, optional): Description
            memory (int, optional): Description
            force_consistent (bool, optional): Description
            use_line_search (bool, optional): Description
        """
        LBFGS.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            memory=memory,
            damping=damping,
            alpha=alpha,
            use_line_search=use_line_search,
            master=None,
            force_consistent=force_consistent,
        )

        self.Hinv = np.linalg.inv(H0_init)
        # print(self.H0)
        # print("THNSDS")
        self.initial = initial
        self.rmsd_dev = rmsd_dev
        self.molindixes = molindixes
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters
        self.fmax_last = None
        self.steps_in_row = 0

    def reset_hessian(self):
        """
        Delete the history of the Hessian
        """
        self.s = []
        self.y = []
        self.rho = []
        self.iteration = 0

    def step(self, f=None):
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it

        Args:
            f (None, optional): Description
        """

        if f is None:
            f = self.atoms.get_forces()

        if self.fmax_last is None:
            self.fmax_last = sqrt((f ** 2).sum(axis=1).max())
        r = self.atoms.get_positions()

        self.update(r, f, self.r0, self.f0)

        s = self.s
        y = self.y
        rho = self.rho
        H0 = self.H0

        loopmax = np.min([self.memory, self.iteration])
        a = np.empty((loopmax,), dtype=np.float64)

        # ## The algorithm itself:
        q = -f.reshape(-1)

        for i in range(loopmax - 1, -1, -1):
            a[i] = rho[i] * np.dot(s[i], q)
            q -= a[i] * y[i]

        # if self.Hinv is not None:
        #     z = np.dot(self.Hinv, q)
        # else:
        #     z = H0 * q

        if self.parameters["calculator"]["preconditioner"]["rmsd_update"][
            "activate"
        ]:
            # self.steps_in_raw+=1
            # or self.steps_in_raw > 50
            current_fmax = sqrt((f ** 2).sum(axis=1).max())
            rmsd = Kabsh_rmsd(self.atoms, self.initial, self.molindixes)
            if self.fmax_last / current_fmax > 3 or rmsd > self.rmsd_dev:
                # # Calculate RMSD between current and initial steps:
                # if Kabsh_rmsd(self.atoms, self.initial, self.molindixes) > self.rmsd_dev:
                print(
                    "################################Applying update",
                    current_fmax,
                    self.steps_in_row,
                    rmsd,
                )
                self.fmax_last = current_fmax
                # self.steps_in_raw = 0
                # Need to pass the inverse Hessian!!!
                self.Hinv = np.linalg.inv(
                    preconditioned_hessian(
                        self.structure,
                        self.fixed_frame,
                        self.parameters,
                        self.atoms,
                        np.eye(3 * len(self.atoms)),
                        task="initial",
                    )
                )

                z = np.dot(self.Hinv, q)

                a0 = self.atoms.copy()
                self.initial = a0
                # self.reset_hessian()
            else:
                z = np.dot(self.Hinv, q)
        else:
            z = np.dot(self.Hinv, q)

        for i in range(loopmax):
            b = rho[i] * np.dot(y[i], z)
            z += s[i] * (a[i] - b)

        self.p = -z.reshape((-1, 3))
        # ##

        g = -f
        if self.use_line_search is True:
            e = self.func(r)
            self.line_search(r, g, e)
            dr = (self.alpha_k * self.p).reshape(len(self.atoms), -1)
        else:
            self.force_calls += 1
            self.function_calls += 1
            dr = self.determine_step(self.p) * self.damping
        self.atoms.set_positions(r + dr)

        self.iteration += 1
        self.r0 = r
        self.f0 = -g
        self.dump(
            (
                self.iteration,
                self.s,
                self.y,
                self.rho,
                self.r0,
                self.f0,
                self.e0,
                self.task,
            )
        )

    def log(self, forces=None):
        """Summary

        Args:
            forces (None, optional): Description

        Returns:
            TYPE: Description
        """
        if self.logfile is None:
            return
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces ** 2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )
        T = time.localtime()
        name = self.__class__.__name__
        w = self.logfile.write
        if self.nsteps == 0:
            w(
                "%s  %4s[%3s] %8s %15s  %12s\n"
                % (" " * len(name), "Step", "FC", "Time", "Energy", "fmax")
            )
            if self.force_consistent:
                w("*Force-consistent energies used in optimization.\n")
        w(
            "%s:  %3d [%3d] %02d:%02d:%02d %15.6f%1s %12.4f\n"
            % (
                name,
                self.nsteps,
                self.force_calls,
                T[3],
                T[4],
                T[5],
                e,
                {1: "*", 0: ""}[self.force_consistent],
                fmax,
            )
        )
        self.logfile.flush()


class TRM_BFGS(BFGS):

    """Summary

    Attributes:
        f0 (TYPE): Description
        fixed_frame (TYPE): Description
        H (TYPE): Description
        H0 (TYPE): Description
        initial (TYPE): Description
        log_accept (bool): Description
        maxstep (TYPE): Description
        molindixes (TYPE): Description
        parameters (TYPE): Description
        r0 (TYPE): Description
        rmsd_dev (TYPE): Description
        structure (TYPE): Description
        tr (TYPE): Description
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=1.0,
        master=None,
        initial=None,
        rmsd_dev=1000.0,
        molindixes=None,
        structure=None,
        H0=None,
        fixed_frame=None,
        parameters=None,
        mu=None,
        A=None,
        known=None,
        tr=1.0,
        eta=0.001,
        r=0.5,
    ):
        """Summary

        Args:
            atoms (TYPE): Description
            restart (None, optional): Description
            logfile (str, optional): Description
            trajectory (None, optional): Description
            maxstep (float, optional): Description
            master (None, optional): Description
            initial (None, optional): Description
            rmsd_dev (float, optional): Description
            molindixes (None, optional): Description
            structure (None, optional): Description
            H0 (None, optional): Description
            fixed_frame (None, optional): Description
            parameters (None, optional): Description
            mu (None, optional): Description
            A (None, optional): Description
            known (None, optional): Description
            tr (float, optional): Description
            eta (float, optional): Description
            r (float, optional): Description
        """
        BFGS.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=None,
        )

        self.H0 = H0  # initial hessian
        self.initial = initial
        self.rmsd_dev = rmsd_dev
        self.molindixes = molindixes
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters
        self.tr = tr
        self.maxstep = maxstep
        self.log_accept = True

        if self.H is None:
            self.H = self.H0

    def update_H(self, dx, df):
        """Input: DX = X -X_old

        DF = F -F_old
        DG = -DF
        H  = hessian
        Task: updated hessian

        Args:
            dx (TYPE): Description
            df (TYPE): Description
        """

        dx = dx[:, np.newaxis]  # dimension nx1
        dx_t = dx.T  # dimension 1xn
        dg = -df[:, np.newaxis]
        dg_t = dg.T

        # JCP, 117,9160. Eq 44
        h1 = np.dot(dg, dg_t)
        h1 = h1 / (np.dot(dg_t, dx))
        h2a = np.dot(self.H, dx)
        h2b = np.dot(dx_t, self.H)
        h2 = np.dot(h2a, h2b)
        h2 = h2 / np.dot(dx_t, h2a)

        self.H += h1 - h2

    def update_BFGS(self, r, f, r0, f0):
        """Summary

        Args:
            r (TYPE): Description
            f (TYPE): Description
            r0 (TYPE): Description
            f0 (TYPE): Description

        Returns:
            TYPE: Description
        """
        if self.H is None:
            self.H = self.H0  # This is Heeian - not inverse!!!
            return
        dr = r.flat - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b

    def step(self, f=None):
        """Summary

        Args:
            f (None, optional): Description
        """
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        f = f.reshape(-1)
        u0 = atoms.get_potential_energy()
        accept = False

        if self.initial:  # Update the Hessian (not inverse!!!)
            # print("Energy", self.atoms.get_potential_energy(), "Force   ",  fmax)
            # # Calculate RMSD between current and initial steps:
            if (
                Kabsh_rmsd(self.atoms, self.initial, self.molindixes)
                > self.rmsd_dev
            ):
                print("################################Applying update")
                self.H = preconditioned_hessian(
                    self.structure,
                    self.fixed_frame,
                    self.parameters,
                    self.atoms,
                    self.H,
                    task="initial",
                )

                a0 = self.atoms.copy()
                self.initial = a0
                self.tr = self.maxstep

        while not accept:
            # Step 1: r - positions, f - forces, self.H - Hessian, tr - trust region
            # Calculate test displacemets
            s = self.min_trm(f, self.H, self.tr)
            atoms.set_positions(r.reshape(-1, 3) + s.reshape(-1, 3))
            # Step 2 Calculate for the found displacemets:
            u = atoms.get_potential_energy()
            f1 = atoms.get_forces().reshape(-1)
            true_gain = u - u0

            y = f - f1
            expected_gain = -np.dot(f, s) + 0.5 * np.dot(s, np.dot(self.H, s))
            harmonic_gain = -0.5 * np.dot(f1, (f - f1))
            # Compute quality:
            s_norm = np.linalg.norm(s)
            quality = true_gain / expected_gain
            f1max = sqrt((f1 ** 2).max())

            true_gain = u - u0
            expected_gain = -np.dot(f, s) + 0.5 * np.dot(s, np.dot(self.H, s))
            harmonic_gain = -0.5 * np.dot(s, (f + f1))

            # Compute quality:
            s_norm = np.linalg.norm(s)

            # if s_norm > 0.02:
            #     quality = true_gain / expected_gain
            # else:
            #     quality = harmonic_gain / expected_gain
            quality = true_gain / expected_gain
            accept = quality > 0.1

            # print("\n")
            # print(s_norm, self.tr)
            # print(true_gain)
            # print(expected_gain)
            # print(harmonic_gain)
            # print("\n")
            # print(accept, quality, self.tr )
            self.log_accept = accept
            if not accept:
                self.log_rejected(forces=f1.reshape(-1, 3))
                atoms.set_positions(r.reshape(-1, 3))

            # Update TrustRadius (self.tr)
            # if quality < 0.25:
            #     self.tr = 0.5 * self.tr
            # elif quality > 0.75 and s_norm > 0.9 * s_norm:
            #     self.tr = 2.0 * self.tr
            #     if self.tr > self.maxstep:
            #         self.tr = self.maxstep

            ll = np.abs(np.dot(s, (y - np.dot(self.H, s))))
            rr = 0.0000001 * s_norm * np.linalg.norm(y - np.dot(self.H, s))

            if ll >= rr:
                print("Update")
                # self.update_H(s, y)
                a = np.dot(
                    (y - np.dot(self.H, s)).reshape(-1, 1),
                    (y - np.dot(self.H, s)).reshape(1, -1),
                )
                b = np.dot(y - np.dot(self.H, s), s)
                self.H += a / b
            else:
                print("Not update")
                pass

            if quality < 0.25:
                self.tr = 0.5 * s_norm
            elif quality > 0.75 and s_norm > 0.9 * self.tr:
                self.tr = 2.0 * self.tr
                if self.tr > self.maxstep:
                    self.tr = self.maxstep

        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        f = f1.copy()
        r = (r.reshape(-1, 3) + s.reshape(-1, 3)).copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))

    def min_trm(self, f, H, tr):
        """Return the minimum of
        E(dx) = -(F * dx + 0.5 * ( dx * H * dx ),
        whithin dx**2 <tr
        IN
        f  = forces        (n,)
        h  = hessian       (nxn)
        tr = trust-radius
        OUT   DX = displacement in cartesian basis
        INTERNAL
        ndim = dimension
        d    = hessian eigenvalues
        w    = hessian eigenvector (in columns)
        g    = gradient in cartesian basis
        gE   = gradient in eigenvector basis
        DX   = displacement in cartesian basis
        DXE  = displacement in eigenvector basis

        Args:
            f (TYPE): Description
            H (TYPE): Description
            tr (TYPE): Description

        Returns:
            TYPE: Description
        """

        # Resize
        ndim = f.size
        shape = f.shape
        f = f.reshape((1, ndim))
        # Diagonalize
        d, w = np.linalg.eigh(H)
        d = d[:, np.newaxis]  # dimension nx1

        gEt = np.dot(f, w)  # Change of basis  ##
        gE = gEt.T  # dimension nx1

        # Count negative,zero,and positive eigenvalues
        neg = (d < -0.0000001).sum()
        zero = (d < 0.0000001).sum() - neg
        # pos = d.size - neg - zero

        # Pull out zero-mode gE
        if zero > 0:
            gE[neg : neg + zero] = np.zeros((zero, 1))

        # Real work start here
        DXE = np.zeros((ndim, 1))

        for i in range(0, ndim):
            if np.absolute(d[i]) > 0.00001:
                DXE[i] = gE[i] / d[i]

        min_d = np.amin(d)

        # Check if h is possitive definite and use trivial result if within trust radius
        if min_d > 0.0:

            if neg != 0:
                print("problem in 'find'!!!")
            if np.linalg.norm(DXE) < tr:
                DX = np.dot(w, DXE)
                DX = DX.reshape(shape)
                return DX

        # If we haven't luck. Let's start with the iteration
        lamb_min = max(0.0, -min_d)
        lamb_max = 1e30
        lamb = min(lamb_min + 0.5, 0.5 * (lamb_min + lamb_max))

        for i in range(0, 100):
            DXE = gE / (d + lamb)
            y = np.sum(DXE ** 2) - tr ** 2
            dy = -2.0 * np.sum((DXE ** 2) / (d + lamb))

            if np.absolute(y / dy) < 0.00001 or np.absolute(y) < 1e-13:
                break

            if y < 0.0:
                lamb_max = min(lamb, lamb_max)
            else:
                lamb_min = max(lamb, lamb_min)

            if dy > 0.0 or lamb_min > lamb_max:
                print("Problem in find. II")

            lamb = lamb - y / dy
            if lamb <= lamb_min or lamb >= lamb_max:
                lamb = 0.5 * (lamb_min + lamb_max)
            # print('iter',i,lamb, lamb_max,lamb_min,y,dy)

        DX = np.dot(w, DXE)
        DX = DX.reshape(shape)
        return DX

    def log(self, forces=None):
        """Summary

        Args:
            forces (None, optional): Description
        """
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces ** 2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )
        accept = self.log_accept
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (
                    " " * len(name),
                    "Step",
                    "Time",
                    "Energy",
                    "fmax",
                    "accepted",
                )
                msg = "%s  %4s %8s %15s %12s %12s\n" % args
                self.logfile.write(msg)

                if self.force_consistent:
                    msg = "*Force-consistent energies used in optimization.\n"
                    self.logfile.write(msg)

            ast = {1: "*", 0: ""}[self.force_consistent]
            args = (
                name,
                self.nsteps,
                T[3],
                T[4],
                T[5],
                e,
                ast,
                fmax,
                self.log_accept,
            )
            msg = "%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f %12s\n" % args
            self.logfile.write(msg)

            self.logfile.flush()

    def log_rejected(self, forces=None):
        """Summary

        Args:
            forces (None, optional): Description
        """
        self.nsteps += 1
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces ** 2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )
        accept = self.log_accept
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (
                    " " * len(name),
                    "Step",
                    "Time",
                    "Energy",
                    "fmax",
                    "accepted",
                )
                msg = "%s  %4s %8s %15s %12s %12s\n" % args
                # self.logfile.write(msg)

                if self.force_consistent:
                    msg = "*Force-consistent energies used in optimization.\n"
                    self.logfile.write(msg)

            ast = {1: "*", 0: ""}[self.force_consistent]
            args = (
                name,
                self.nsteps,
                T[3],
                T[4],
                T[5],
                e,
                ast,
                fmax,
                self.log_accept,
            )
            msg = "%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f %12s\n" % args
            self.logfile.write(msg)

            self.logfile.flush()


class TRM_BFGS_IPI(BFGS):

    """Summary

    Attributes:
        f0 (TYPE): Description
        fixed_frame (TYPE): Description
        H (TYPE): Description
        H0 (TYPE): Description
        initial (TYPE): Description
        lastforce (TYPE): Description
        log_accept (bool): Description
        maxstep (TYPE): Description
        molindixes (TYPE): Description
        parameters (TYPE): Description
        r0 (TYPE): Description
        restart (TYPE): Description
        rmsd_dev (TYPE): Description
        steps (int): Description
        structure (TYPE): Description
        tr (TYPE): Description
        tr_init (TYPE): Description
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=0.15,
        master=None,
        initial=None,
        rmsd_dev=1000.0,
        molindixes=None,
        structure=None,
        H0=None,
        fixed_frame=None,
        parameters=None,
        mu=None,
        A=None,
        known=None,
        tr=0.04,
        eta=0.001,
        r=0.5,
    ):
        """Summary

        Args:
            atoms (TYPE): Description
            restart (None, optional): Description
            logfile (str, optional): Description
            trajectory (None, optional): Description
            maxstep (float, optional): Description
            master (None, optional): Description
            initial (None, optional): Description
            rmsd_dev (float, optional): Description
            molindixes (None, optional): Description
            structure (None, optional): Description
            H0 (None, optional): Description
            fixed_frame (None, optional): Description
            parameters (None, optional): Description
            mu (None, optional): Description
            A (None, optional): Description
            known (None, optional): Description
            tr (float, optional): Description
            eta (float, optional): Description
            r (float, optional): Description
        """
        BFGS.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=None,
        )

        BOHR_to_angstr = 0.52917721  # in AA
        HARTREE_to_eV = 27.211383  # in eV

        self.H0 = H0  # initial hessian
        self.initial = initial
        self.rmsd_dev = rmsd_dev
        self.molindixes = molindixes
        self.structure = structure
        self.fixed_frame = fixed_frame
        self.parameters = parameters
        self.tr = tr
        self.tr_init = tr
        self.maxstep = maxstep
        self.log_accept = True
        self.steps = 0
        self.lastforce = None
        self.restart = restart

        if self.H is None:
            self.H = self.H0
            # self.H=np.eye(3*len(self.atoms))*100

    def update_H(self, dx, df):
        """Update the Hessian matrix

        Args:
            dx {array}: displacements x -x_old
            df {array}: force gradient f -f_old
        """

        dx = dx[:, np.newaxis]  # dimension nx1
        dx_t = dx.T  # dimension 1xn
        dg = -df[:, np.newaxis]
        dg_t = dg.T

        # JCP, 117,9160. Eq 44
        h1 = np.dot(dg, dg_t)
        h1 = h1 / (np.dot(dg_t, dx))
        h2a = np.dot(self.H, dx)
        h2b = np.dot(dx_t, self.H)
        h2 = np.dot(h2a, h2b)
        h2 = h2 / np.dot(dx_t, h2a)

        self.H += h1 - h2

    def step(self, f=None):
        """Summary

        Args:
            f (None, optional): Description
        """
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()
            self.lastforce = sqrt((f ** 2).sum(axis=1).max())

        r = atoms.get_positions()
        # print("Energy")
        current = sqrt((f ** 2).sum(axis=1).max())
        f = f.reshape(-1)
        # print(f)

        u0 = atoms.get_potential_energy()

        self.steps += 1

        if (
            Kabsh_rmsd(self.atoms, self.initial, self.molindixes)
            > self.rmsd_dev
            and self.parameters["calculator"]["preconditioner"]["rmsd_update"][
                "activate"
            ]
        ):
            # if self.lastforce / current < 2.7:
            print("################################Applying update")
            self.H = preconditioned_hessian(
                self.structure,
                self.fixed_frame,
                self.parameters,
                self.atoms,
                self.H,
                task="update",
            )

            a0 = self.atoms.copy()
            self.initial = a0
            self.steps = 0
            self.lastforce = current
            self.tr = self.tr_init

        accept = False
        rejected_steps = 0
        while not accept:
            # Step 1: r - positions, f - forces, self.H - Hessian, tr - trust region
            # Calculate test displacemets
            # print(f)
            # print(self.tr)
            s = self.min_trm(f, self.H, self.tr)
            # print("Displacements")
            # print(s.reshape(-1,3))
            # sys.exit(0)
            atoms.set_positions(r.reshape(-1, 3) + s.reshape(-1, 3))
            # print(r.reshape(-1,3) + s.reshape(-1,3))
            # sys.exit(0)
            # Step 2 Calculate for the found displacemets:
            u = atoms.get_potential_energy()
            f1 = atoms.get_forces().reshape(
                -1
            )  # Already correct sign! f1 = -grad(u)

            true_gain = u - u0
            expected_gain = -np.dot(f, s) + 0.5 * np.dot(s, np.dot(self.H, s))

            # Compute quality:
            s_norm = np.linalg.norm(s)
            quality = true_gain / expected_gain
            accept = quality > 0.1

            # Update TrustRadius (self.tr)
            if quality < 0.25:
                self.tr = 0.25 * s_norm
                # if self.tr < 0.0001:
                # self.tr = self.maxstep
            elif quality > 0.75 and s_norm == self.tr:
                self.tr = 2.0 * self.tr
                if self.tr > self.maxstep:
                    self.tr = self.maxstep
            print("######### self.tr {}".format(self.tr))

            # print(accept, quality, self.tr )
            self.log_accept = accept
            if not accept:
                rejected_steps += 1
                self.log_rejected(forces=f1.reshape(-1, 3))
                atoms.set_positions(r.reshape(-1, 3))
                if rejected_steps == 10:
                    # reset preconditioner
                    self.H = np.eye(3 * len(self.atoms)) * 10
                    # self.H = preconditioned_hessian(
                    # self.structure,
                    # self.fixed_frame,
                    # self.parameters,
                    # self.atoms,
                    # self.H,
                    # task="update",
                    # )

                    a0 = self.atoms.copy()
                    self.initial = a0
                    self.steps = 0
                    self.lastforce = current
                    self.tr = self.tr_init
            else:
                rejected_steps = 0

        y = np.subtract(f1, f)
        # print(y)
        # self.update_BFGS(r.reshape(-1,3) + s.reshape(-1,3), f1, r, f)
        self.update_H(s.flatten(), y.flatten())
        # sys.exit(0)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        f = f1.copy()
        r = (r.reshape(-1, 3) + s.reshape(-1, 3)).copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))

    def min_trm(self, f, h, tr):
        """Return the minimum of
        E(dx) = -(F * dx + 0.5 * ( dx * H * dx ),
        whithin dx**2 <tr

        INTERNAL
            ndim = dimension
            d    = hessian eigenvalues
            w    = hessian eigenvector (in columns)
            g    = gradient in cartesian basis
            gE   = gradient in eigenvector basis
            DX   = displacement in cartesian basis
            DXE  = displacement in eigenvector basis

        Args:
            f (array): forces
            h (matrix): Hessian
            tr (float): trust-radius

        Returns:
            DX: displacement in cartesian basis
        """

        # Resize
        ndim = f.size
        shape = f.shape
        f = f.reshape((1, ndim))

        # Diagonalize
        d, w = np.linalg.eigh(h)
        d = d[:, np.newaxis]  # dimension nx1

        gEt = np.dot(f, w)  # Change of basis  ##
        gE = gEt.T  # dimension nx1

        # Count negative,zero,and positive eigenvalues
        neg = (d < -0.0000001).sum()
        zero = (d < 0.0000001).sum() - neg
        # pos = d.size - neg - zero

        # Pull out zero-mode gE
        if zero > 0:
            gE[neg : neg + zero] = np.zeros((zero, 1))

        # Real work start here
        DXE = np.zeros((ndim, 1))

        for i in range(0, ndim):
            if np.absolute(d[i]) > 0.00001:
                DXE[i] = gE[i] / d[i]

        min_d = np.amin(d)

        # Check if h is possitive definite and use trivial
        # result if within trust radius
        if min_d > 0.0:

            if neg != 0:
                print("problem in 'find'!!!")
            if np.linalg.norm(DXE) < tr:
                DX = np.dot(w, DXE)
                DX = DX.reshape(shape)
                return DX

        # If we haven't luck. Let's start with the iteration
        lamb_min = max(0.0, -min_d)
        lamb_max = 1e30
        lamb = min(lamb_min + 0.5, 0.5 * (lamb_min + lamb_max))

        for i in range(0, 100):
            DXE = gE / (d + lamb)
            y = np.sum(DXE ** 2) - tr ** 2
            dy = -2.0 * np.sum((DXE ** 2) / (d + lamb))

            if np.absolute(y / dy) < 0.00001 or np.absolute(y) < 1e-13:
                break

            if y < 0.0:
                lamb_max = min(lamb, lamb_max)
            else:
                lamb_min = max(lamb, lamb_min)

            if dy > 0.0 or lamb_min > lamb_max:
                print("Problem in find. II")

            lamb = lamb - y / dy
            if lamb <= lamb_min or lamb >= lamb_max:
                lamb = 0.5 * (lamb_min + lamb_max)
        #  print 'iter',i,lamb, lamb_max,lamb_min,y,dy

        DX = np.dot(w, DXE)
        DX = DX.reshape(shape)
        return DX

    def log(self, forces=None):
        """Summary

        Args:
            forces (None, optional): Description
        """
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces ** 2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )
        accept = self.log_accept
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (
                    " " * len(name),
                    "Step",
                    "Time",
                    "Energy",
                    "fmax",
                    "accepted",
                )
                msg = "%s  %4s %8s %15s %12s %12s\n" % args
                self.logfile.write(msg)

                if self.force_consistent:
                    msg = "*Force-consistent energies used in optimization.\n"
                    self.logfile.write(msg)

            ast = {1: "*", 0: ""}[self.force_consistent]
            args = (
                name,
                self.nsteps,
                T[3],
                T[4],
                T[5],
                e,
                ast,
                fmax,
                self.log_accept,
            )
            msg = "%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f %12s\n" % args
            self.logfile.write(msg)

            self.logfile.flush()

    def log_rejected(self, forces=None):
        """Summary

        Args:
            forces (None, optional): Description
        """
        self.nsteps += 1
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces ** 2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )
        accept = self.log_accept
        T = time.localtime()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (
                    " " * len(name),
                    "Step",
                    "Time",
                    "Energy",
                    "fmax",
                    "accepted",
                )
                msg = "%s  %4s %8s %15s %12s %12s\n" % args
                # self.logfile.write(msg)

                if self.force_consistent:
                    msg = "*Force-consistent energies used in optimization.\n"
                    self.logfile.write(msg)

            ast = {1: "*", 0: ""}[self.force_consistent]
            args = (
                name,
                self.nsteps,
                T[3],
                T[4],
                T[5],
                e,
                ast,
                fmax,
                self.log_accept,
            )
            msg = "%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f %12s\n" % args
            self.logfile.write(msg)

            self.logfile.flush()
