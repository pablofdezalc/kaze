% KAZE.m
% Class for testing KAZE with the vlbenchmark
% Author: Pablo F. Alcantarilla
% Date: 27-12-2015
% Note: The code is an adaptation of the original code from Karel Lenc
%**************************************************************************
classdef KAZE < localFeatures.GenericLocalFeatureExtractor & ...
    helpers.GenericInstaller

properties (SetAccess=public, GetAccess=public)
    Opts = struct(...
      'soffset', 1.30,...        % the base scale offset (sigma units)
      'omax', 4,...              % the coarsest nonlinear scale space level (sigma units)
      'nsublevels', 4,...        % number of sublevels per octave
      'dthreshold', .0008,...    % Feature detector threshold response for accepting points
      'descriptor', 5,...        % Descriptor Type (See KAZEConfig.h for more info)
      'sderivatives', 1,...      % Gaussian smoothing for the derivatives     
      'diffusivity', 1,...       % Diffusivity type: 0->PM_G1, 1->PM_G2, 2->Weickert, 3->Charbonnier
      'show_results', 1,...      % 1 in case we want to visualize the detection results
      'save_scale_space', 0 ...  % 1 in case we want to save the nonlinear scale space images. 0 otherwise
      );
  end
  
  properties (Constant,Hidden)
    RootInstallDir = fullfile('data','software','kaze','');
    SrcDir = fullfile(localFeatures.Kaze.RootInstallDir,'');
    BinPath = fullfile(localFeatures.Kaze.SrcDir,'kaze_features');
  end

  methods
    function obj = KAZE(varargin)
      import helpers.*;
      obj.Name = 'KAZE';
      obj.ExtractsDescriptors = true;
      varargin = obj.checkInstall(varargin);
      varargin = obj.configureLogger(obj.Name,varargin);
      obj.Opts = vl_argparse(obj.Opts, varargin);
    end

    function [frames descriptors] = extractFeatures(obj, origImagePath)
      import helpers.*;
      import localFeatures.helpers.*;
      frames = obj.loadFeatures(origImagePath,true);
      if numel(frames) > 0; return; end;      
      obj.info('Computing features of image %s.',...
        getFileName(origImagePath));
      [imagePath imIsTmp] = obj.ensureImageFormat(origImagePath);
      tmpName = tempname;
      outFeaturesFile = [tmpName '.kaze'];
      args = obj.buildArgs(imagePath, outFeaturesFile);
      cmd = [obj.BinPath ' ' args];
      obj.debug('Executing: %s',cmd);
      startTime = tic;
      [status,msg] = helpers.osExec('.',cmd,'-echo');
      if status ~= 0
        error('%d: %s: %s', status, cmd, msg) ;
      end
      timeElapsed = toc(startTime);
      [frames descriptors] = ...
        readFeaturesFile(outFeaturesFile,'FloatDesc',true);
      delete(outFeaturesFile);
      if imIsTmp, delete(imagePath); end;
      obj.debug('%d features from image %s computed in %gs',...
        size(frames,2),getFileName(imagePath),timeElapsed);
      obj.storeFeatures(imagePath, frames, []);
    end

    function sign = getSignature(obj)
      sign = [helpers.fileSignature(obj.BinPath) ';'...
              helpers.struct2str(obj.Opts)];
    end
  end

  methods (Access=protected)
    function args = buildArgs(obj, imagePath, outFile)
      % -nl - do not write the hessian keypoint type
      args = sprintf(' "%s" --output "%s"',...
        imagePath, outFile);
      fields = fieldnames(obj.Opts);
      for i = 1:numel(fields)
        val = obj.Opts.(fields{i});
        if ~isempty(val)
          args = [args,' --',fields{i},' ', num2str(val)];
        end
      end
    end

    function deps = getDependencies(obj)
      deps = {helpers.Installer()};
    end

    function res = isCompiled(obj)
      res = exist(obj.BinPath,'file');
    end

    function compile(obj)
      % Run cmake; make and make install
      import helpers.*;
      if obj.isCompiled()
        return;
      end
      if ~exist(obj.SrcDir,'dir')
        error('Source code of KAZE feature detector not present in %s.',...
          obj.SrcDir);
      end

      fprintf('Compiling KAZE\n');
      % Run cmake with sys. libraries environment
      [status msg] = helpers.osExec(obj.SrcDir,'cmake .','-echo');
      if status ~= 0
        error('CMake error: \n%s',msg);
      end
      % Run Make
      [status msg] = helpers.osExec(obj.SrcDir,'make','-echo');
      if status ~= 0
        error('Make error: \n%s',msg);
      end
    end
  end
end
