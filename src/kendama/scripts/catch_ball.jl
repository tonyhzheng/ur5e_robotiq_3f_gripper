#!/usr/bin/env julia
# This node uses MPC algorithms to calculate the optimal inputs that minimizes
# the distance between the BARC and the reference trajectory. For this demonstration
# a dummy reference trajectory is fed into this node which then publishes the optimal
# state trajectory.

println("Initializing Packages...")

using RobotOS
@rosimport std_msgs.msg: Float64MultiArray, Float64
# @rosimport barc.msg: Input, barc_state
@rosimport data_service.msg: TimeData
@rosimport geometry_msgs.msg: Vector3
rostypegen()
# using barc.msg
using data_service.msg
using geometry_msgs.msg
using std_msgs.msg
using JuMP
using Ipopt


println("DONE")

# -------------------------------------------------------------------
#                           MPC PARAMETERS
# -------------------------------------------------------------------

println("Defining Parameters...")

# Number of states / Number of inputs
# 1: End-effector, 2: Ball
nq = 8 # x1, z1, dx1, dz1, x2, z2, dx2, dz2
nu = 2



count = 0
floatrarray = Float64MultiArray()
#### MAKE MPC TERMS PARAMETERS
# Horizons
# n = get_param("/numStepsToLookAhead")           # MPC Horizon (CFTOC Horizon)
n = 10
dt = 0.05
r = 0.43
g = 9.81
mass_hand = 0.279
mass_ball = 0.083
string_length = 0.43

# Box constraints on input, A*u <= B
qmax =  [.8;  1	; 	2 ;  3 ; 1000; 1000;1000;1000]
qmin = [-.3; -0.3   ;  -2 ; -3;-1000;-1000;-1000;-1000]
# qmax =  [1;1; 5 ; 5 ; 1000; 1000;1000;1000]
# qmin = [-1; -1  ; -5; -5;-1000;-1000;-1000;-1000]
umax = [20;20]
umin = -umax

# x1_0 = 0.0
# z1_0 = 0.0
# dx1_0 = 0.0
# dz1_0 = 0.0

# x2_0 = 0.0
# z2_0 = 0.0
# dx2_0 = 0.0
# dz2_0 = 0.0


println("DONE")

# -------------------------------------------------------------------
#                         SET UP CFTOC MODEL
# -------------------------------------------------------------------

println("Creating CFTOC Model...")

m = Model(solver = IpoptSolver(print_level=0))

# Variables
# @variable(m,z[1:nz,1:n+1])              # State vectors (nz x n+1)
@variable(m,q[1:nq,1:n+1])              # State vectors (nz x n+1)
@variable(m,u[1:nu,1:n])                # Input vectors (nu x n)

# Objective Function
cost = 0
for i in 1:n+1
	cost = cost + (q[1,i] - q[5,i])^2 + (q[4,i])^2 
end
@objective(m, :Min, cost)

@NLparameter(m, x1_0 == 0); 
@NLparameter(m, z1_0 == 0); 
@NLparameter(m, dx1_0 == 0); 
@NLparameter(m, dz1_0 == 0); 
@NLparameter(m, x2_0 == 0); 
@NLparameter(m, z2_0 == 0); 
@NLparameter(m, dx2_0 == 0); 
@NLparameter(m, dz2_0 == 0); 

# State Constraints
for i in 1:n
	@constraint(m, q[1,i+1]<=qmax[1])
	@constraint(m, q[1,i+1]>=qmin[1])
	@constraint(m, q[2,i+1]<=qmax[2])
	@constraint(m, q[2,i+1]>=qmin[2])
	@constraint(m, q[3,i+1]<=qmax[3])
	@constraint(m, q[3,i+1]>=qmin[3])
	@constraint(m, q[4,i+1]<=qmax[4])
	@constraint(m, q[4,i+1]>=qmin[4])
	# @constraint(m, q[5,i+1]<=qmax[5])
	# @constraint(m, q[5,i+1]>=qmin[5])
	# @constraint(m, q[6,i+1]<=qmax[6])
	# @constraint(m, q[6,i+1]>=qmin[7])
	@constraint(m, u[1,i]<=umax[1])
	@constraint(m, u[1,i]>=umin[1])
	@constraint(m, u[2,i]<=umax[2])
	@constraint(m, u[2,i]>=umin[2])
end

# Initial Condition and Constraints
# x1, z1, dx1, dz1, x2, z2, dx2, dz2
@NLconstraint(m, q[1,1]==x1_0)
@NLconstraint(m, q[2,1]==z1_0)
@NLconstraint(m, q[3,1]==dx1_0)
@NLconstraint(m, q[4,1]==dz1_0)

@NLconstraint(m, q[5,1]==x2_0)
@NLconstraint(m, q[6,1]==z2_0)
@NLconstraint(m, q[7,1]==dx2_0)
@NLconstraint(m, q[8,1]==dz2_0)

# Dynamics Constraints
for i in 1:n
	@constraint(m, q[1,i+1]==q[1,i]+dt*q[3,i])
	@constraint(m, q[2,i+1]==q[2,i]+dt*q[4,i])
	@constraint(m, q[5,i+1]==q[5,i]+dt*q[7,i])
	@constraint(m, q[6,i+1]==q[6,i]+dt*q[8,i])

	@constraint(m, q[3,i+1]==q[3,i]+dt*(u[1,i])/(mass_hand))
	@constraint(m, q[4,i+1]==q[4,i]+dt*(u[2,i]-mass_hand*g)/(mass_hand))
	@constraint(m, q[7,i+1]==q[7,i])
	@constraint(m, q[8,i+1]==q[8,i]+dt*(-g))
	# Linearized Model
	# @constraint(m, q[:,i+1]==q[:,i]+dt*(A*q[:,i]+B*u[:,i])
end

# String constraint
# for i in 1:n
# 	@NLconstraint(m, ((q[1,i]-q[5,i])^2 +(q[2,i]-q[6,i])^2)^2 <= (string_length)^2)
# end

println("DONE")

# -------------------------------------------------------------------
#                             INITIAL SOLVE
# -------------------------------------------------------------------

println("Initial solve...")
solve(m)
println("DONE")

#-------------------------------------------------------------------
#                             ROS FUNCTIONS
# -------------------------------------------------------------------

println("Running MPC...")

function hand_state_callback(msg::Float64MultiArray)
	setvalue(x1_0, msg.data[1])
	setvalue(z1_0, msg.data[3])
	setvalue(dx1_0, msg.data[4])
	setvalue(dz1_0, msg.data[6])
end

function ball_state_callback(msg::Float64MultiArray)
	# print(msg)
	setvalue(x2_0, msg.data[1])
	setvalue(z2_0, msg.data[3])
	setvalue(dx2_0, msg.data[4])
	setvalue(dz2_0, msg.data[6])
end

function loop(pub_one)
    global count,floatrarray
    while ! is_shutdown()
	loop_rate = Rate(60)
	count = count+1
        # -----------------------------------------------------------
        #                      RUNNING MPC
        # -----------------------------------------------------------
		if count>1
	    # toc()
		end

		# print(getvalue(x1_0))
		# print(getvalue(x2_0))
        # tic()
        solve(m)                                # Solve CFTOC
        qOpt = getvalue(q)

        uOpt = getvalue(u)

		x_hand_opt = qOpt[1,2]
		z_hand_opt = qOpt[2,2]
		xdot_hand_opt = qOpt[3,2]
		zdot_hand_opt = qOpt[4,2]

        # Publish optimal inputs and optimal state trajectory
        # Only need to publish state traj for this demonstration
		floatrarray.data = [x_hand_opt, z_hand_opt, xdot_hand_opt, zdot_hand_opt, uOpt[1,1],uOpt[2,1]]
		# hand_state = Float64MultiArray([x_hand_opt, z_hand_opt, xdot_hand_opt, zdot_hand_opt])
		publish(pub_one, floatrarray)

        rossleep(loop_rate)
    end
end

function main()
    init_node("MPC_node")
	global count,floatrarray
    hand_state_pub = Publisher("hand_state_opt", Float64MultiArray, queue_size=1)
	hand_state_sub = Subscriber("hand_state", Float64MultiArray, hand_state_callback, queue_size=1)
	ball_state_sub = Subscriber("ball_state", Float64MultiArray, ball_state_callback, queue_size=1)
    loop(hand_state_pub)
end

if ! isinteractive()
    main()
end
