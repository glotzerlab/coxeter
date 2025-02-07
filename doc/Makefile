# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
STATICDIR     = _static

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# Note:
#	Second line ensures that there is a build x3d folder
# 	Third line copies the X3D files from _static (Sphinx doesn't copy them correctly)
#	Fourth line runs the sphinx build command
%: Makefile
	@echo "Creating ./$(BUILDDIR)/$@/$(STATICDIR)/x3d..."; 
	@mkdir -p "$(BUILDDIR)/$@/$(STATICDIR)/x3d"
	@cp -r "./$(SOURCEDIR)/$(STATICDIR)/x3d" "./$(BUILDDIR)/$@/$(STATICDIR)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
