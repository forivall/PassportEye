from os import path
import distutils.cmd
import json

class MakePackageJsonCommand(distutils.cmd.Command):
    description = 'Create a package.json file for this python module'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        distmeta = self.distribution.metadata
        meta = {}
        for attr in distmeta._METHOD_BASENAMES:
            if attr in ('long_description','author_email','maintainer_email'):
                continue
            value = None
            if hasattr(distmeta, attr):
                value = getattr(distmeta, attr)
            if value is None:
                continue

            if attr == 'name':
                value = value.lower()
            if attr == 'keywords':
                try:
                    value = [item for s in value for item in s.split()]
                except:
                    pass

            if attr in ('author', 'maintainer') and hasattr(distmeta, attr + '_email'):
                email = getattr(distmeta, attr + '_email')
                if email is not None:
                    meta[attr] = value + ' <' + email + '>'
                else:
                    meta[attr] = value
            else:
                meta[attr] = value

        with open(path.join(path.dirname(__file__), 'package.json'), 'w') as f:
            json.dump(meta, f, indent='  ')
            f.write('\n')
