function [x, mu, z, iter] = qpintpoint2(Q,F,c,d)
% Método de Puntos Interiores para el problema cuadrático convexo
%       Min (1/2)*x'*Q*x + c'*x
%       s.a.      F*x >= d
%
% Q es nxn simétrica positiva definida
% F es pxn matriz de desigualdades
% c es un vector columna de tamaño n
% d es un vector columna de tamaño p
%
%        Optimización Numérica
%               ITAM
%       21 de septiembre de 2023
%---------------------------------------------------------------

tol = 1e-05;    % tolerancia a las condiciones necesarias de primer orden
                % para un mínimo local
maxiter = 200;  % número máximo de iteraciones permitidas

iter = 0;
gamma = 0.5;

n = length(c);  % número de variables
p = length(d);  % número de restricciones de desigualdad

x = ones(n,1);  % punto inicial
mu = ones(p,1);
z = ones(p,1);

% Función de la cuál queremos encontrar el cero
G = [Q*x-F'*mu+c;
     -F*x+z+d;
     mu.*z];

% vector para graficación
cnpo = [];
norma = norm(G);
disp('Iter     CNPO')
while(norma > tol && iter < maxiter)
    JacobianG = [   Q        -F'        zeros(n,p);
                   -F      zeros(p)    eye(p);
              zeros(p,n)   diag(z)     diag(mu);
                    ]; % Matriz Jacobiana de G
    
    ld = -[Q*x-F'*mu+c;
           -F*x+z+d;
           mu.*z-gamma*ones(p,1)]; % G_gamma

    % Solución del sistema lineal
    Delta = JacobianG\ld;
    % Partición en las variables x, mu, z
    Deltax = Delta(1:n);
    Deltamu = Delta(n+1:n+p);
    Deltaz = Delta(n+p+1:end);

    alfamu = recorta(mu,Deltamu);
    alfaz = recorta(z,Deltaz);
    
    alfa = min(alfamu, alfaz);

    x = x + alfa*Deltax;
    mu = mu + alfa*Deltamu;
    z = z + alfa*Deltaz;

    G = [ Q*x+-(F'*mu)+c;
         -F*x+z+d;
          mu.*z];

    iter = iter +1;
    gamma = gamma/2;
    norma = norm(G);
    cnpo = [cnpo norma];
    fprintf('%3.0f %2.8f \n', iter, norma)
end

figure;
semilogy([1:iter], cnpo)
xlabel('Número de iteraciones')
ylabel('CNPO')
title('Convergencia del método de puntos interiores')
end

function [alfa] = recorta(u,v)
% Recorta la longitud del vector v tal que
%       u + alfa*v >= 0
%       donde u > 0.
% Rutina para puntos interiores.

n = length(v);
valfa = ones(n,1);
for k = 1:n
    if (v(k) <0)
        valfa(k) = min(1, -(u(k)/v(k)));
    end
end

alfa = (0.995)*min(valfa);

end