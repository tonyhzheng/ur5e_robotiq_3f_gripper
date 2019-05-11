#This function uses the unit vector along axis of joint in base
#configuration, positions of points you choose on each joint axis, type
#joint (R or P), configuration of the manipulator, and joint rate vectors
#to solve the direct velocity kinematics of an open chain manipulator. The
#output will be manipulator Jacobian and spatial + angular velocity of the
#tool frame.

import numpy as np

def manipJac(Ro,Po,w,Q,J,theta,thetadot):

#Ro = reference configuration of the tool frame WRT base frame
#Po = configuration of the manipulator
#w = unit vectors along the axis of the joints
#Q = positions on each joint axis
#J = type of joints, 1 is revolute, 2 is prismatic. ex. [1 1 2] RRP
#theta = list of angles to rotate each joint by

Gst0 = np.array([Ro Po; 0,0,0 1])
Gst0 = np.array([Ro,Po],[0,0,0,1])
eXiTheta=eye(4);

#Initial Gst(0) = [R(0) P(0); 0 1]
#R(0) is the starting orientation of the end effector
#P(0) is the starting position of the end effector


#xi = twist of coordinates
#for each additional joint, the 

for i=1:numel(w)/3
    if J(i) == 1
        xi = [cross((-1*w(:,:,i)),Q(:,:,i)); w(:,:,i)];
    elseif J(i) == 2
        v = w(:,:,i);
        xi = [v; 0;0;0];
    end
    xilist(:,:,i) = xi;
    wHat=[0 -xi(6) xi(5);
        xi(6) 0 -xi(4);
        -xi(5) xi(4) 0 ];
    norm(w(:,:,i))
    if (norm(w(:,:,i))~=0)&&J(i)==1
        eW = eye(3)+wHat*sin(theta(i))+(1-cos(theta(i)))*wHat^2;
        T(:,:,i)=[eW ((eye(3)-eW)*wHat+w(:,:,i)*w(:,:,i)'*theta(i))*xi(1:3);0 0 0 1];
    else 
        
        eW = eye(3)+(wHat/norm(w(1:3)))*sin(norm(w(1:3))*theta(i))+(1-cos(norm(w(1:3))*theta(i)))*((wHat^2)/(norm(w(1:3))^2));
        T(:,:,i)=[eW xi(1:3)*theta(i);0 0 0 1];
        
    end
    #T=[eW (eye(3)-eW)*(cross(xi(4:6),xi(1:3)))+xi(4:6)*xi(4:6)'*xi(1:3)*theta(i);0 0 0 1]   
    
    
    R=eW(1:3,1:3);
    I=R*R';
    I2=R'*R;
    I3=det(R);
    #e = exp(wHat(:,:,i)*theta(i));
    #Gst=Gst*exp(wHat(:,:,i)*theta(i));
    eXiTheta = eXiTheta*T(:,:,i);
    
    
end
#{
for i=numel(w)/3:-1:1
    Gst = Gst*T(:,:,i)
end
#}
xilist
T
Gst=eXiTheta*Gst0


Jacobian=[xilist(:,:,1)];#first twist not affected by movement
#Compute the Jacobian
for i=2:numel(w)/3
    g1_iminus1=eye(4);
    for j=1:i-1
        g1_iminus1 = g1_iminus1*T(:,:,j);
    end
    p = g1_iminus1(1:3,4);
    phat = [0 -p(3) p(2);p(3) 0 -p(1);-p(2) p(1) 0];
    Ad(1:3,1:3) = g1_iminus1(1:3,1:3);
    Ad(1:3,4:6) = phat*g1_iminus1(1:3,1:3);
    Ad(4:6,1:3) = 0;
    Ad(4:6,4:6) = g1_iminus1(1:3,1:3);
    Ad;
    xiprime = Ad*xilist(:,:,i);
    Jacobian(:,i)=xiprime;
end
Jacobian

#Compute the velocity
Vs = Jacobian*thetadot

p=Gst(1:3,4);
phat = [0 -p(3) p(2);p(3) 0 -p(1);-p(2) p(1) 0];
Ad(1:3,1:3) = Gst(1:3,1:3);
Ad(1:3,4:6) = phat*Gst(1:3,1:3);
Ad(4:6,1:3) = 0;
Ad(4:6,4:6) = Gst(1:3,1:3);
Vb = Ad^-1*Vs

end