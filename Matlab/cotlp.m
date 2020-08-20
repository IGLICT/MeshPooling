function [v, f, n, L, M, VV, CotWeight, Laplace_Matrix, L_unweight] = cotlp(filename, K)
% read obj file and output some information about mesh
% v: vertices
% f: faces
% n: vertices normals
% L: laplace matrix
% M: no use
% VV: adjacency list
% CotWeight: cotweight matrix
% Laplace_Matrix: transfer vertices to edges with cotweight
% L_unweight: transfer vertices to edges without cotweight

if nargin == 1
    K = 3;
end
[v, f, n, II, JJ, SS, AA, vv, cotweight, laplace_matrix, a, a, L_unweight] = meshlp(filename, K);
v = v';
n = n';
W=sparse(II, JJ, SS);
L=W;
A=AA;
Atmp = sparse(1:length(A),1:length(A),1./A);
M=sparse(1:length(A),1:length(A),A);
%L = sparse(diag(1./ A)) * W;
% L = Atmp * W;
VV=vv;
CotWeight=cotweight';
Laplace_Matrix=laplace_matrix';
L_unweight=L_unweight';
end

