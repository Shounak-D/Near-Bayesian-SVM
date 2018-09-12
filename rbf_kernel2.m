function kval = rbf_kernel2(u,v)
%RBF_KERNEL Radial basis function kernel for SVM functions
global rbf_sigma

if rbf_sigma > 0
    kval = exp(-(1/(2*rbf_sigma^2))*(repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
    -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1)));
elseif rbf_sigma==-1
    kval = u * v';
else
    kval = (u * v' + 1).^( -1 * rbf_sigma );
end

if issparse(kval)
    kval = full(kval);
end