% Kalman Filter simulation
% btremaine@gmail.com   9/23/2022
%
% Formulate as m-script to show details of implementation
% R2022A

% There are three blocks running in the simulation
% 1. 'noisy' plant, disturbance noise wd & sensor noise vn 
% 2. 'ideal' plant as comparison of 'truth'
% 3. Standard Kalman Filter
% 4. Drive input u[k] with arbitrary discrete signal

% First define c.t. system then discretize 
% define plant
% Parameters defining the system
m = 100.0;              % system inertia
k =  0.01;              % spring constant
b =  1.0;               % damping constant

% We have measurement of noisy position only, not velocity.
% Kalman Filter will provide estimates of position and velocity.

% Continuous-time System matrices
A = [0, 1. ; -k/m -b/m];
B = [0; 1/m];
C = [1.0, 0.0];         % position
D = 0;
sys = ss(A, B, C, D);

% plot c.t. step response10Np= floor(200/Ts);
Ts = 1.0;
Np= 50;
%
close ALL, clear ALL

[yc, t] = step(sys,Np*Ts);
plot(t,yc); figure(1); title('unit step response')

% convert to noiseless discrete system using sample time Ts
d_sys = c2d(sys,Ts);
% ideal plant, no noise
Fk = d_sys.A;
Bk = d_sys.B;
Hk = d_sys.C;
Dd = d_sys.D;
% disturbance noise into torque input
Bp = Bk;

% Augment plant model with noise inputs
% x[k+1] = A*x[k] + B1*u[k] + B2*wd[k]
% y[k]   = C*x[k] + D*u[k] + vn[k]

% ----------------------------------------------------------------------
% Start simulation loop, process each time step, Ts
% initialize arrays for processing
n= size(Fk,1);
r= size(Hk,1);
m= size(Bk,2);

x= zeros(n,Np);
x_hat= zeros(n,Np);
x_bar= zeros(n,Np);
y= zeros(r,Np);
y_hat= zeros(r,Np);
%
Ppr= zeros(n);
Pupd= zeros(n);
M= zeros(n);
L= zeros(n,Np);
% ideal plant
xp= zeros(n,Np);
yp= zeros(r,Np);

% add pre-generated noise to system
% process noise is at torque input
% R is rxr
wd = 0.02*randn(1,Np); wd = wd - mean(wd); % process
Qk = cov(wd);
%
vn = 3.60*randn(1,Np); vn = vn - mean(vn);  % sensor noise <try 3.60>
Rk = cov(vn'); % 1x1

% define input
u = ones(1,Np);
u(1,1) = 0.0;

for k= 2:Np
    % Update noisy plant at time k
    xp(:,k) = Fk*xp(:,k-1) + Bk*u(:,k-1) + Bp*wd(:,k);
    yp(:,k) = Hk*xp(:,k-1) + Dd*u(:,k-1) + vn(k);
    %
    % Update ideal plant at time k (noiseless) for reference only
    x(:,k) = Fk*x(:,k-1) + Bk*u(:,k-1);
    y(:,k) = Hk*x(:,k-1) + Dd*u(:,k-1);
    %%  +++++++++++++ Standard Kalman Filter +++++++++++++++++++
    % get input noise and disturbance
    %  --- no control used, wd & vn pre-generated
    % x_bar == predicted update,   x_bar_k|k-1
    % x_hat == meaurement update,  
    %
    % predicted i.e. model time update    
    x_bar(:,k) = Fk*x_hat(:,k-1);    % no control, Stengel 4.3-12 time update
    Ppre= Fk*Pupd*Fk' + Qk;          % P_k|k-1     Stengel 4.3-13 time update

    % measurement update
    Sk = Hk*Ppre*Hk' + Rk;
    Kk = Ppre*Hk'*(eye(size(Sk))\Sk);            % Stengel 4.3-14 observation update
    x_hat(:,k) = x_bar(:,k) + Kk*(yp(:,k) - Hk*x_bar(:,k)); % Stengal 4.3-15
    Pupd =  (eye(size(Kk,1)) - Kk*Hk)*Ppre;      % Stengel 4.3-16 Cov estimate update
    y_hat(:,k) = Hk*x_hat(:,k)     ;             % no direct feed-through
    L(:,k) = Kk;  % save for plotting
 % +++++++++++++ end Kalman Flter ++++++++++++++++++++++++
%%
end

% plot results
% states:
figure(2);
subplot(211)
plot(x(1,:)); hold on 
plot(x_bar(1,:));
plot(xp(1,:),'.'); 
plot(x_hat(1,:))
title('X1'); legend('xideal', 'xbar', 'xp','xhat')

subplot(212)
plot(x(2,:)); hold on
plot(x_bar(2,:));
plot(xp(2,:),'.'); 
plot(x_hat(2,:))
title('X2'); legend('xideal', 'xbar', 'xp','xhat')
hold off

% outputs:
figure(3)
plot(y) ;hold on
plot(yp,'.')
plot(y_hat)
title('output'); legend('yideal', 'ynoisy', 'yhat')
hold off

% L gains
figure(4); 
subplot(211)
plot(L(1,:)); title('L1')
subplot(212)
plot(L(2,:)); title('L2')

%% RMSE 
err_plant= std((yp-y));
RMSE1= err_plant

err_y_hat= std((y-y_hat));
RMSE2= err_y_hat

