% Extended Kalman Filter simulation
% btremaine@gmail.com   9/24/2022
%
% Formulate as m-script to show details of EKF implementation
% R2022A

% There are three blocks run in this simulation
% 1. 'noisy' plant, disturbance noise wd & sensor noise vn 
% 2. 'ideal' plant as comparison of 'truth'
% 3. Extended Kalman Filter
% 4. There is no u[k] input with this 2D Radar model.

close ALL; clear

%% Define system as a radar receiving 2D information of a target
%  assuming constant velocity
% in the dimensions x for horizontal and y for vertical distances
% The distance to the target is r = sqrt(x^2 + y^2.
% the angle to the target is theta, the states are [x, x_dot, y y_dot]'
% The definition of the discrete system equations are:
% dx1/dt = x2
% dx3/dt = x4
% 
%         | 1 dt 0  0 |
%  A    = | 0  1 0  0 |
%         | 0  0 1 dt |
%         | 0  0 0  1 |
%
%  observation model is z= [r ; th]
%                        = [sqrt(x^2 + y^2) ; tam^-1(y/x)]
%
%  Therefore, dynamic matrix is linear but observation matrix is nonlinear
%

% States
% x1 = x
% x2 = x_dot
% x3 = y
% x4 = y_dot
%
% state-space model 
%  A as above for discrete model
%  B == 0, no input
%  input: none
% ==============================================================
%% Calculate system without noise and plot
tp=(0:0.1:50);
Np= length(tp);
Vt= 20;          % ft/sec at attack angle alpha
vn= zeros(2,Np);
alpha=15.0*pi/180; % attack angle, rad/sec
%% initial pos & vel
x0=[100;Vt*cos(alpha);1000;Vt*sin(alpha)];
[xr,yr] = Radar2D(x0,tp,vn);

s=(1:Np);
figure(1); % plot ideal systen states without noise.
subplot(211)
plot(s,xr(1,:), s,xr(3,:)); title('true noiseless position states')
legend('x', 'y'); 
subplot(212)
plot(s,xr(2,:), s,xr(4,:) ); title('true noiseless velocity states')
legend('x_d','y_d');

% add pre-generated noise to system
% R is rxr
n = 4;
r = 2;
% Sensor noise, vn & Rk
%
vn(1,:) = 10*randn(1,Np); 
vn(2,:) = (2*pi/180)*randn(1,Np);
vn(1,:) = vn(1,:) - mean(vn(1,:));
vn(2,:) = vn(2,:) - mean(vn(2,:));
Rk =  diag([2^2, (2*pi/180)^2]) ;         % Covariance of sensor noise 
% Process noise, wd & Qk
%
wd = 1E-1*randn(1,Np); % not used this model
wd = wd - mean(wd);    %
Qk = diag([0.001, 0.02, 0.001, 0.02]);    % Covariance of process noise
%
[xrn,yrn] = Radar2D(x0,tp,vn);
%
figure(2)
subplot(211)
plot(yrn(1,:)'); title('noisy plant range, ra')
subplot(212)
plot(yrn(2,:)'); title('noisy plant angle, theta')
%
x_hat= zeros(n,Np);
x_bar= zeros(n,Np);
y_hat= zeros(r,Np);
%
L1= zeros(n,Np);
L2= zeros(n,Np);
% ideal plant
yp= yrn; % noisy y for Kalman input

%% Run Kalman Filter
% =====================================================================
% convert to noiseless discrete system using sample time Ts= dt
dt = tp(2)-tp(1);

% Augment plant model with noise inputs
% x[k+1] = A*x[k] + B1*u[k] + B2*wd[k]
% y[k]   = C*x[k] + D*u[k] + vn[k]
% ----------------------------------------------------------------------
% Start simulation loop, process each time step, Ts
% linearize plant in Extended Kalman Filter
% Dynamics are linear but observation is nonlinear
% state-space model
% x1 = hoizontal distance
% dx1/dt = x2
% x3 = vertical distance
% dx3/dt = x4
% 
%         | 1 dt 0  0 |
%  F    = | 0  1 0  0 |
%         | 0  0 1 dt |
%         | 0  0 0  1 |
%
%  observation model is z= [r ; th]  == [sqrt(x1^2 + x3^2) ; atan(x3/x1)]
%
%  Jacobian of observation matrix:
%   DH = [  cos(th)   0 sin(th)   0;
%          -sin(th)/r 0 cos(th)/r 0];
Fk = [1 dt 0 0; 0 1 0 0 ; 0 0 1 dt; 0 0 0 1];     
Bk= zeros(4,1);
D= zeros(4,1);   % Dk not used in this model
% Hk observation is non-linear & time varying
% for k=1
%    initialize starting point
  x_hat(:,1)= x0;
  yp(:,1)=yp(:,2); % get rid of (0,0)
  Pupd= diag([0.06, 0.20, 200, 2.0]);
% end
for k= 2:Np
    x1= yp(1,k-1);
    x3= yp(2,k-1);
    th = atan2(x3,x1);
    ra = sqrt(x1^2 + x3^2)^0.5;
    % Calculate Jacobian each k
    Hk=  [ cos(th) 0 sin(th) 0;
          -sin(th)/ra 0 cos(th)/ra 0];

    %%  +++++++++++++ Linearized Kalman Filter +++++++++++++
    % get input noise and disturbance
    %  --- no control used, wd & vn pre-generated
    % x_bar == predicted update,   x_bar_k|k-1
    % x_hat == meaurement update,  
    %
    % predicted i.e. model time update    
    x_bar(:,k) = Fk*x_hat(:,k-1);    % no control, Stengel 4.3-12 time update
    Ppre= Fk*Pupd*Fk' + Qk;          % P_k|k-1  Stengel 4.3-13 time update

    % measurement update
    Sk = Hk*Ppre*Hk' + Rk;
    Kk = Ppre*Hk'*(eye(size(Sk))*inv(Sk));       % Stengel 4.3-14 observation update
    hk = Observation(x_bar(1,k), x_bar(3,k));    % returns (range, theta) 
    x_hat(:,k) = x_bar(:,k) + Kk*(yp(:,k) - hk); % Stengal 4.3-15
    Pupd =  (eye(size(Kk,1)) - Kk*Hk)*Ppre; % Stengel 4.3-16 Cov estimate update
    % +++++++++++++ end Extended Kalman Flter ++++++++++++++++++++++++
    % save for plotting
    hk = Observation(x_hat(1,k), x_hat(3,k));
    y_hat(:,k) = hk;        % no direct feed-through
    L1(:,k) = Kk(:,1);
    L2(:,k) = Kk(:,2);
    %
end
%% plot results
xp= xrn; % plant states for plotting
y= yr;   % ideal y for plotting
s=(1:Np);
% states:
figure(3);
subplot(211)
plot(s,xrn(1,s),'r',s,xp(1,s),'--',s,x_bar(1,s),'g',s,x_hat(1,s),'k');
title('X1'); legend('xideal', 'xnoisy', 'xbar','xhat')
subplot(212)
plot(s,xrn(2,s),'r',s,xp(2,s),'--',s,x_bar(2,s),'g',s,x_hat(2,s),'k');
title('X2'); legend('xideal', 'xnoisy', 'xbar','xhat')

% outputs:
figure(4)
subplot(211)
plot(s,y(1,:),'r',s,yp(1,:),'--',s,y_hat(1,:),'b') ;
title('output range, ra'); legend('yideal', 'ynoisy', 'yhat','location', 'southeast');
subplot(212)
plot(s,y(2,:),'r',s,yp(2,:),'--',s,y_hat(2,:),'b') ;
title('output,  theta'); legend('yideal', 'ynoisy', 'yhat','location','southeast');

% L gains
figure(5); 
subplot(211);
plot(L1'); title('L1')  % check if plotting correct, n gains at each k
subplot(212)
plot(L2'); title('L2')

% plot path
figure(6)
plot(xr(1,:),xr(3,:),'x'); title('target path')
hold on
plot(xp(1,:),xp(3,:),'+')
plot(x_hat(1,:),x_hat(3,:),'o')
plot(x_bar(1,:),x_bar(3,:),'-'); legend('xr','xp','xhat','xbar','Location','southeast')
axis equal


%% RMSE 
err_plant= std((yp(1,:)-y(1,:)));
RMSE1= err_plant

err_y_hat= std((y(1,:)-y_hat(1,:)));
RMSE2= err_y_hat


%% Functions
function [xr,yr] = Radar2D(x0,tp,noise)
% state-space model
% x1 = hoizontal distance, x
% dx1/dt = x2
% x3 = vertical distance, y
% dx3/dt = x4
% assumes constant target velocity
% 
%         | 1 dt 0  0 |
%  A    = | 0  1 0  0 |
%         | 0  0 1 dt |
%         | 0  0 0  1 |
%
%  observation model is z= [r ; th]
%                        = [sqrt(x1^2 + x3^2) ; tan^-1(x3/x1)]
% ==============================================================
n = length(x0);
Np= length(tp);
xr = zeros(n,Np);
yr= zeros(2,Np);
dt= tp(2)-tp(1);
xr(:,1) = x0;
for k=2:Np
 xr(1,k) = xr(1,k-1) + dt * xr(2,k-1);
 xr(2,k) = xr(2,k-1);
 xr(3,k) = xr(3,k-1) + dt * xr(4,k-1);
 xr(4,k) = xr(4,k-1);
 %
 yr(1,k) = sqrt(xr(1,k)^2 + xr(3,k)^2) + noise(1,k);
 yr(2,k) = atan2(xr(3,k), xr(1,k)) + noise(2,k);
end
end

function [h_k] = Observation(x1, x3)
% Calculate non-linear observation
% x1 = x position
% x3 = y psoition
% h_h= [ range, angle ]
h_k = [sqrt(x1^2 + x3^2) ; atan2(x3,x1)]; 
end
