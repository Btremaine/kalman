% Unscented Kalman Filter simulation
% btremaine@gmail.com   10/6/2022
%
% Formulate as m-script to show details of UKF implementation
% R2022A

% There are three blocks run in this simulation
% 1. 'noisy' plant, disturbance noise wd & sensor noise vn 
% 2. 'ideal' plant as comparison of 'truth'
% 3. Unscented Kalman Filter
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
% States:
%  x1 = x
%  x2 = x_dot
%  x3 = y
%  x4 = y_dot
%
%  Jacobian of observation matrix:
%   DH = [  cos(th)   0 sin(th)   0;44
%          -sin(th)/r 0 cos(th)/r 0];
% =========================================================================
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
% Sensor noise, vn & Rk
%
vn(1,:) = 10*randn(1,Np); 
vn(2,:) = (2*pi/180)*randn(1,Np);
vn(1,:) = vn(1,:) - mean(vn(1,:));
vn(2,:) = vn(2,:) - mean(vn(2,:));
Rk =  diag([2^2, (2*pi/180)^2]) ;        % sensor noise variance  
% Process noise, wd & Qk
%
wd = 1E-1*randn(1,Np); % not used this model
wd = wd - mean(wd);    %
Qk = diag([0.001, 0.02, 0.001, 0.02]) ;  % process noise variance 
%
[xrn,yrn] = Radar2D(x0,tp,vn);
%
figure(2)
subplot(211)
plot(yrn(1,:)'); title('noisy plant range, ra')
subplot(212)
plot(yrn(2,:)'); title('noisy plant angle, theta')
%
n = 4;
r = 2;
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
Fk = [1 dt 0 0; 0 1 0 0 ; 0 0 1 dt; 0 0 0 1];     
Bk= zeros(4,1);
D= zeros(4,1);   % Dk not used in this model
%
f= @(x,dt)[x(1)+dt*x(2); x(2); x(3)+dt*x(4); x(4)];    % state equations
h= @(x)[sqrt(x(1)^2 + x(3)^2) ; atan2(x(3),x(1))];     % output function
%    initialize starting point
x_hat(:,1)= x0;
yp(:,1)=yp(:,2); % get rid of (0,0)
Px = diag([0.050 0.170 198.0 2.5]); % init from 1st run
%
% weights for UT
[Wm,Wc,c] = weights(length(Fk));
%
for k= 2:Np
    x1= yp(1,k-1);     % noisy measurement
    x3= yp(2,k-1);     % noisy measurement
    th = atan2(x3,x1); % noisy measurement
    ra = sqrt(x1^2 + x3^2)^0.5;
    %%  +++++++++++++ Unscented Kalman Filter ++++++++++++++++++++++++++++
    % get input noise and disturbance
    %  --- no control used, wd & vn pre-generated
    % x_bar == predicted update,   x_bar_k|k-1
    % x_hat == meaurement update,  
    %
    % *********** time update; prediction ***************************
    % Can use Fk instead of sigma points because state transition is
    % linear, but using f(x) for clarity here. 
    x_bar(:,k) = f(x_hat(:,k-1), dt);   % state update, Stengel 4.3-12
    Px_pre= Fk*Px*Fk' + Qk;             % P_k|k-1  Stengel 4.3-13 time update X
    %
    % *********** measurement update ********************************
    % Need to use sigma points for nonlinear sensor
    % 1. -- compute X, 2n+1 sigma points 
    % 2. -- compute Y, using hk evaluated at each sigma point
    % 3. -- compute z, the weighted mean of Y
    % 4. -- compute S,  cov(Y-z) + Rk
    % 5. -- computed Pxy, cov((X-mu_pre),(Y-z)  ??? check out mu
    % 6. -- compute Kk,  T.*inv(S)

    [X, xm]= sigmas(x_bar(:,k),Px_pre,c,Wm,Wc);   % 1. Measurement sigma points
                                                  %    and predicted state mean
    [Y, zm] = CalcZatXi(X, Wm);                   % 2. calculate zk at sigma points
                                                  %    and predicted measurement mean                             
    Sk = (Y-zm)*diag(Wc)*(Y-zm)' + Rk;            % 4. Py, predicted meas. covariance
    Pxy= (X-xm)*diag(Wc)*(Y-zm)';                 % 5. Pxy, cross-covariance, nxr
    Kk = Pxy/Sk;                                  % 6. Kalman gain (nxr)
    %
    x_hat(:,k) = x_bar(:,k) + Kk*(yp(:,k) - zm);  % Stengal 4.3-15 innovation
    Px = Px_pre - Kk*Sk*Kk';             
    
    % ++++++++++++++++ end Unscented Kalman Flter +++++++++++++++++++++++++++++++
    % save data for plotting ...
    y_hat(:,k) = h(x_hat(:,k));         % direct nonlinear calculation
    L1(:,k) = Kk(:,1);
    L2(:,k) = Kk(:,2); 
    %
end % loop
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
plot(s,y(1,:),'r',s,yp(1,:),'--',s,y_hat(1,:),'b+') ;
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

% polar plot
figure(7)
polarplot(yp(2,:),yp(1,:),'.'); hold
polarplot(y_hat(2,:),y_hat(1,:),'-'); axis([0 90 0 1800]);
title('measured & filtered - UKF')

%% RMSE 
err_plant= std((yp(1,:)-y(1,:)));
RMSE1= err_plant;
disp(RMSE1);

err_y_hat= std((y(1,:)-y_hat(1,:)));
RMSE2= err_y_hat;
disp(RMSE2);

%% All helper Functions placed below this line
%
% Calculate plant with noise over time vector
function [xr,yr] = Radar2D(x0,tp,noise)
% state-space model
% initial state:  x0
% time vector:    tp
% Gaussian noise: noise
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
%%
function [Z, zm] = CalcZatXi(X, W)
% calculate sensor h(x) at each sigma point, Xi
% uses h() := [range ; angle]
h= @(x)[sqrt(x(1)^2 + x(3)^2) ; atan2(x(3),x(1))];     % output function
m = length(X);
Z = zeros(2,m);
for j=1:m
    Z(:,j) = h(X(:,j));
end
zm = sum(Z.*W,2);
end
%%
function [X, mu]=sigmas(x,P,c,Wm,Wc)
%Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       c: coefficient
%Output:
%       X: Sigma points
A = c*chol(P)';
Y = x(:,ones(1,numel(x)));
X = [x Y+A Y-A];
mu = sum(X.*Wm,2);
end
%% 
function [Wm, Wc, c] = weights(L)
% ref:
% "The Unscented Kalman Filter for Nonlinear Estimation"
%  Eric A. Wan and Rudolph van der Merwe
% L: # of states
% Wm: weights for mean
% Wc: weights for covariance
alpha= sqrt(2);                             % default, tunable
ki=0;                                       % default, tunable
beta=1.0;                                   % default, tunable
lambda=alpha^2*(L+ki)-L;                    % scaling factor
c = sqrt(L+lambda);                         % coeff for sigma
Wm= zeros(1,2*L+1);
Wc= zeros(1,2*L+1);
Wm(1) = lambda/(L+lambda);
Wc(1) = lambda/(L+lambda) + (1-alpha^2+beta);
for k=2:2*L+1
  Wm(k) = 1/(2*(L+lambda));
  Wc(k) = 1/(2*(L+lambda));
end
end
