//
// Created by Michael Plumaris on 24/01/2022.
//

#include "tudat/astro/relativity/ComptonModel.h"
#include "tudat/astro/relativity/metric.h"

namespace tudat{



// Michael
Eigen::Vector3d computeComptonRelativisticAcceleration(
        const Eigen::Vector3d& vectorToAcceleratedBody,
        const double gravitationalParameter,
        const double ComptonWavelength )
{
    double distance = vectorToAcceleratedBody.norm( );
    return gravitationalParameter * vectorToAcceleratedBody
    / (distance * 2 * ComptonWavelength * ComptonWavelength) ;
}

/*
//! Compute gravitational force.
Eigen::Vector3d computeGravitationalForce(
    const double universalGravitationalParameter,
    const double massOfBodySubjectToForce,
    const Eigen::Vector3d& positionOfBodySubjectToForce,
    const double massOfBodyExertingForce,
    const Eigen::Vector3d& positionOfBodyExertingForce )
{
    return massOfBodySubjectToForce * computeGravitationalAcceleration(
        universalGravitationalParameter, positionOfBodySubjectToForce,
        massOfBodyExertingForce, positionOfBodyExertingForce );
}

*/

}
