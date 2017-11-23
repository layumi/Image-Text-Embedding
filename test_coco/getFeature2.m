function x = getFeature2(net,oim,im_mean,inputname,outputname)
if(~isempty(im_mean))
    im = bsxfun(@minus,single(oim),im_mean);
else
    im = single(oim);
end
net.vars(net.getVarIndex(outputname)).precious = true;
net.eval({inputname,gpuArray(im)}) ;
x = gather(net.vars(net.getVarIndex(outputname)).value);
end

