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



#### MAKE MPC TERMS PARAMETERS
# Horizons
# n = get_param("/numStepsToLookAhead")           # MPC Horizon (CFTOC Horizon)
n = 30
dt = 0.025
r = 0.43
g = 9.81
mass_hand = 0.279
mass_ball = 0.083
string_length = 0.43

# Box constraints on input, A*u <= B
qmax =  [1 ;0.75; 3 ; 3 ; 1000; 1000;1000;1000]
qmin = -[-1; 0  ; -3; -3;-1000;-1000;-1000;-1000]
umax = [20;20]
umin = -umax

x1_0 = 0
z1_0 = 0
dx1_0 = 0
dz1_0 = 0

x2_0 = 0
z2_0 = 0
dx2_0 = 0
dz2_0 = 0

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
	cost = cost + norm(q[1,i] - q[5,i])
end
@objective(m, :Min, cost)

# State Constraints
for i in 1:n
	@constraint(m, q[:,i+1]<=qmax)
	@constraint(m, q[:,i+1]>=qmin)
	@constraint(m, u[:,i]<=umax)
	@constraint(m, u[:,i]>=umin)
end

# Initial Condition and Constraints
# x1, z1, dx1, dz1, x2, z2, dx2, dz2
@constraint(m, q[1,1]==x1_0)
@constraint(m, q[2,1]==z1_0)
@constraint(m, q[3,1]==dx1_0)
@constraint(m, q[4,1]==dz1_0)

@constraint(m, q[5,1]==x2_0)
@constraint(m, q[6,1]==z2_0)
@constraint(m, q[7,1]==dx2_0)
@constraint(m, q[8,1]==dz2_0)

# Dynamics Constraints
for i in 1:n
	@constraint(m, q[1,i+1]==q[1,i]+dt*q[3,i])
	@constraint(m, q[2,i+1]==q[2,i]+dt*q[4,i])
	@constraint(m, q[5,i+1]==q[5,i]+dt*q[7,i])
	@constraint(m, q[6,i+1]==q[6,i]+dt*q[8,i])

	@constraint(m, q[3,i+1]==q[3,i]+dt*(u[1,i])/(mass_hand)
	@constraint(m, q[4,i+1]==q[4,i]+dt*(u[2,i]-mass_hand*g)/(mass_hand)
	@constraint(m, q[7,i+1]==q[7,i])
	@constraint(m, q[8,i+1]==q[8,i]+dt*(-g))
	# Linearized Model
	# @constraint(m, q[:,i+1]==q[:,i]+dt*(A*q[:,i]+B*u[:,i])
end

# String constraint
for i in 1:n
	@constraint(m, norm([q[1,i]-q[5,i], q[2,i]-q[6,i]]) <= string_length-0.05)
end

println("DONE")

# -------------------------------------------------------------------
#                             INITIAL SOLVE
# -------------------------------------------------------------------

println("Initial solve...")
solve(m)
println("DONE")

-------------------------------------------------------------------
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
	setvalue(x2_0, msg.data[1])
	setvalue(z2_0, msg.data[3])
	setvalue(dx2_0, msg.data[4])
	setvalue(dz2_0, msg.data[6])
end

function loop(pub_one, pub_two)
    global count
    while ! is_shutdown()
	loop_rate = Rate(get_param("/loop_rate"))
	count = count+1
        # -----------------------------------------------------------
        #                      RUNNING MPC
        # -----------------------------------------------------------
		if count>1
	    toc()
		end
        tic()
        solve(m)                                # Solve CFTOC
        qOpt = getvalue(q)

		x_hand_opt = qOpt[1,2]
		z_hand_opt = qOpt[2,2]
		xdot_hand_opt = qOpt[3,2]
		zdot_hand_opt = qOpt[4,2]

        # Publish optimal inputs and optimal state trajectory
        # Only need to publish state traj for this demonstration

		hand_state = Float64MultiArray(x_hand_opt, z_hand_opt, xdot_hand_opt, zdot_hand_opt)
		publish(hand_state_pub, hand_state)

        rossleep(loop_rate)
    end
end

function main()
    init_node("MPC_node")
	global count
    hand_state_pub = Publisher("hand_state_opt", Float64MultiArray, queue_size=1)
	hand_state_sub = Subscriber("hand_state", Float64MultiArray, hand_state_callback, queue_size=1)
	ball_state_sub = Subscriber("ball_state", Float64MultiArray, ball_state_callback, queue_size=1)
    loop(pub1, pub2)
end

if ! isinteractive()
    main()
end
