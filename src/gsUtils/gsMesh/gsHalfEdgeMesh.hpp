/** @file gsHalfEdgeMesh.hpp

    @brief Provides implementation of the gsHalfEdgeMesh class.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): L. Groiss, J. Vogl
*/

#pragma once

namespace gismo
{
//**********************************************
//************ class gsHalfEdgeMesh ************
//**********************************************
struct less_than_ptr
{
    bool operator()(gsMesh<>::gsVertexHandle lhs, gsMesh<>::gsVertexHandle rhs)
    {
        return ((*lhs) < (*rhs));
    }
};

struct equal_ptr
{
    bool operator()(gsMesh<>::gsVertexHandle lhs, gsMesh<>::gsVertexHandle rhs)
    {
        return ((*lhs) == (*rhs));
    }
};

//********************************************************************************

template<class T>
gsHalfEdgeMesh<T>::gsHalfEdgeMesh(const gsMesh<> &mesh)
    : gsMesh<>(mesh)
{
    std::sort(this->vertex.begin(), this->vertex.end(), less_than_ptr());
    std::vector<gsVertex<double> *, std::allocator<gsVertex<double> *> >::iterator
    last = std::unique(this->vertex.begin(), this->vertex.end(), equal_ptr());
    this->vertex.erase(last, this->vertex.end());
    for (std::size_t i = 0; i < this->face.size(); i++)
    {
        m_halfedges.push_back(getInternHalfedge(this->face[i], 1));
        m_halfedges.push_back(getInternHalfedge(this->face[i], 2));
        m_halfedges.push_back(getInternHalfedge(this->face[i], 3));
    }
    m_boundary = Boundary(m_halfedges);
    m_n = this->vertex.size() - m_boundary.getNumberOfVertices();
    sortVertices();
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::getNumberOfVertices() const
{
    return this->vertex.size();
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::getNumberOfTriangles() const
{
    return this->face.size();
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::getNumberOfInnerVertices() const
{
    return m_n;
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::getNumberOfBoundaryVertices() const
{
    return m_boundary.getNumberOfVertices();
}

template<class T>
const gsMesh<>::gsVertexHandle &gsHalfEdgeMesh<T>::getVertex(const std::size_t vertexIndex) const
{
    if (vertexIndex > this->vertex.size())
    {
        std::cerr << "Error: [" << __PRETTY_FUNCTION__ << "] Vertex with index 'vertexIndex'=" << vertexIndex
                  << " does not exist. There are only " << this->vertex.size() << " vertices." << std::endl;
    }
    return ((this->vertex[m_sorting[vertexIndex - 1]]));
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::getVertexIndex(const gsMesh<>::gsVertexHandle &vertex) const
{
    return m_inverseSorting[getInternVertexIndex(vertex)];
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::getGlobalVertexIndex(std::size_t localVertexIndex, std::size_t triangleIndex) const
{
    if ((localVertexIndex != 1 && localVertexIndex != 2 && localVertexIndex != 3)
        || triangleIndex > getNumberOfTriangles() - 1)
        std::cerr << "Error: [" << __PRETTY_FUNCTION__ << "] The 'localVertexIndex'=" << localVertexIndex
                  << " should be 1,2 or 3 and the triangle with 'triangleIndex'=" << triangleIndex << " does not exist"
                  << std::endl;
    if (localVertexIndex == 1)
        return getVertexIndex((this->face[triangleIndex]->vertices[0]));
    if (localVertexIndex == 2)
        return getVertexIndex((this->face[triangleIndex]->vertices[1]));
    return getVertexIndex((this->face[triangleIndex]->vertices[2]));
}

template<class T>
double gsHalfEdgeMesh<T>::getBoundaryLength() const
{
    return m_boundary.getLength();
}

template<class T>
bool rangeCheck(const std::vector<int> &corners, const std::size_t minimum, const std::size_t maximum)
{
    for (std::vector<int>::const_iterator it = corners.begin(); it != corners.end(); it++)
    {
        if (*it < minimum || *it > maximum)
        { return false; }
    }
    return true;
}

template<class T>
std::vector<double> gsHalfEdgeMesh<T>::getCornerLengths(std::vector<int> &corners) const
{
    std::size_t B = getNumberOfBoundaryVertices();
    if (!rangeCheck<T>(corners, 1, B))
    {
        std::cerr << "Error: [" << __PRETTY_FUNCTION__ << "] The corners must be <= number of boundary vertices."
                  << std::endl;
        std::cerr << "One of these is >= " << getNumberOfBoundaryVertices() << std::endl;
        for (std::vector<int>::const_iterator it = corners.begin(); it != corners.end(); it++)
        {
            std::cout << *it << std::endl;
        }
    }
    std::sort(corners.begin(), corners.end());
    std::size_t s = corners.size();
    std::vector<double> lengths;
    for (std::size_t i = 0; i < s; i++)
    {
        lengths.push_back(m_boundary.getDistanceBetween(corners[i], corners[(i + 1) % s]));
    }
    return lengths;
}

template<class T>
double gsHalfEdgeMesh<T>::getShortestBoundaryDistanceBetween(std::size_t i, std::size_t j) const
{
    return m_boundary.getShortestDistanceBetween(i, j);
}

template<class T>
const std::vector<double> gsHalfEdgeMesh<T>::getBoundaryChordLengths() const
{
    return m_boundary.getHalfedgeLengths();
}

template<class T>
double gsHalfEdgeMesh<T>::getHalfedgeLength(std::size_t originVertexIndex, std::size_t endVertexIndex) const
{
    if (originVertexIndex > this->vertex.size() || endVertexIndex > this->vertex.size())
    {
        std::cerr << "Error: [" << __PRETTY_FUNCTION__ << "] One of the input vertex indices " << originVertexIndex
                  << " or " << endVertexIndex << " does not exist. There are only " << this->vertex.size()
                  << " vertices." << std::endl;
    }
    return gsVector3d<real_t>(getVertex(originVertexIndex)->x() - getVertex(endVertexIndex)->x(),
                                     getVertex(originVertexIndex)->y() - getVertex(endVertexIndex)->y(),
                                     getVertex(originVertexIndex)->z() - getVertex(endVertexIndex)->z()).norm();
}

template<class T>
triangleVertexIndex gsHalfEdgeMesh<T>::isTriangleVertex(std::size_t vertexIndex, std::size_t triangleIndex) const
{
    if (vertexIndex > this->vertex.size())
    {
        std::cerr << "Warning: [" << __PRETTY_FUNCTION__ << "] Vertex with vertex index " << vertexIndex
                  << " does not exist. There are only " << this->vertex.size() << " vertices." << std::endl;
        return error;
    }
    if (triangleIndex > getNumberOfTriangles())
    {
        std::cerr << "Warning: [" << __PRETTY_FUNCTION__ << "] The " << triangleIndex
                  << "-th triangle does not exist. There are only " << getNumberOfTriangles() << " triangles."
                  << std::endl;
        return error;
    }
    if (*(this->vertex[m_sorting[vertexIndex - 1]]) == *(this->face[triangleIndex]->vertices[0]))
    { return first; }
    if (*(this->vertex[m_sorting[vertexIndex - 1]]) == *(this->face[triangleIndex]->vertices[1]))
    { return second; }
    if (*(this->vertex[m_sorting[vertexIndex - 1]]) == *(this->face[triangleIndex]->vertices[2]))
    { return third; }
    return error;
}

template<class T>
const std::queue<typename gsHalfEdgeMesh<T>::Boundary::Chain::Halfedge>
gsHalfEdgeMesh<T>::getOppositeHalfedges(const std::size_t vertexIndex, const bool innerVertex) const
{
    std::queue<typename Boundary::Chain::Halfedge> oppositeHalfedges;
    if (vertexIndex > this->vertex.size())
    {
        std::cerr << "Error: [" << __PRETTY_FUNCTION__ << "] The vertex with index " << vertexIndex
                  << " does not exist. There are only " << this->vertex.size() << " vertices." << std::endl;
        return oppositeHalfedges;
    }
    else if (vertexIndex > m_n && innerVertex)
    {
        std::cerr << "Warning: [" << __PRETTY_FUNCTION__ << "] Inner vertex with index 'vertexIndex' = " << vertexIndex
                  << "is not an inner vertex. There are only " << m_n << " inner vertices." << std::endl;
    }

    for (std::size_t i = 0; i < getNumberOfTriangles(); i++)
    {
        switch (isTriangleVertex(vertexIndex, i))
        {
            case first:
                oppositeHalfedges.push(typename Boundary::Chain::Halfedge(getGlobalVertexIndex(3, i),
                                                                 getGlobalVertexIndex(2, i),
                                                                 getHalfedgeLength(getGlobalVertexIndex(3, i),
                                                                                   getGlobalVertexIndex(2, i))));
                break;
            case second:
                oppositeHalfedges.push(typename Boundary::Chain::Halfedge(getGlobalVertexIndex(1, i),
                                                                 getGlobalVertexIndex(3, i),
                                                                 getHalfedgeLength(getGlobalVertexIndex(1, i),
                                                                                   getGlobalVertexIndex(3, i))));
                break;
            case third:
                oppositeHalfedges.push(typename Boundary::Chain::Halfedge(getGlobalVertexIndex(2, i),
                                                                 getGlobalVertexIndex(1, i),
                                                                 getHalfedgeLength(getGlobalVertexIndex(2, i),
                                                                                   getGlobalVertexIndex(1, i))));
                break;
            default:
                //not supposed to show up
                break;
        }
    }
    return oppositeHalfedges;
}

//*****************************************************************************************************
//*****************************************************************************************************
//*******************THE******INTERN******FUNCTIONS******ARE******NOW******FOLLOWING*******************
//*****************************************************************************************************
//*****************************************************************************************************
template<class T>
bool gsHalfEdgeMesh<T>::isBoundaryVertex(const std::size_t internVertexIndex) const
{
    if (internVertexIndex > this->vertex.size() - 1)
    {
        std::cerr << "Warning: [" << __PRETTY_FUNCTION__ << "] Vertex with intern vertex index = " << internVertexIndex
                  << " does not exist. There are only " << this->vertex.size() << " vertices." << std::endl;
        return false;
    }
    else
        return m_boundary.isVertexContained(internVertexIndex);
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::getInternVertexIndex(const gsMesh<real_t>::gsVertexHandle &vertex) const
{
    std::size_t internVertexIndex = 0;
    for (std::size_t i = 0; i < this->vertex.size(); i++)
    {
        if ((*(this->vertex[i])) == *vertex)
            return internVertexIndex;
        internVertexIndex++;
    }
    if (internVertexIndex > this->vertex.size() - 1)
    {
        std::cerr << "Warning: [" << __PRETTY_FUNCTION__ << "] The ESS_IO::IO_Vertex 'vertex' = (" << vertex->x()
                  << ", " << vertex->y() << ", " << vertex->z() << ") is not contained in gsHalfEdgeMesh vertices"
                  << std::endl;
        return 0;
    }
    return 0;
}

template<class T>
const typename gsHalfEdgeMesh<T>::Boundary::Chain::Halfedge
gsHalfEdgeMesh<T>::getInternHalfedge(const gsMesh<real_t>::gsFaceHandle &triangle, std::size_t numberOfHalfedge) const
{
    std::size_t index1 = getInternVertexIndex((triangle->vertices[0]));
    std::size_t index2 = getInternVertexIndex((triangle->vertices[1]));
    std::size_t index3 = getInternVertexIndex((triangle->vertices[2]));
    if (numberOfHalfedge < 1 || numberOfHalfedge > 3)
    {
        std::cerr << "Warning: [" << __PRETTY_FUNCTION__ << "] The inputted number of the halfedge " << numberOfHalfedge
                  << "  is supposed to be 1,2 or 3. Because input was not expected, first halfedge is returned."
                  << std::endl;
        numberOfHalfedge = 1;
    }
    if (numberOfHalfedge == 1)
    {
        return typename Boundary::Chain::Halfedge(index2, index1,
                                         gsVector3d<real_t>(
                                             triangle->vertices[1]->x() - triangle->vertices[0]->x(),
                                             triangle->vertices[1]->y() - triangle->vertices[0]->y(),
                                             triangle->vertices[1]->z() - triangle->vertices[0]->z()).norm());
    }
    if (numberOfHalfedge == 2)
    {
        return typename Boundary::Chain::Halfedge(index3, index2,
                                         gsVector3d<real_t>(
                                             triangle->vertices[2]->x() - triangle->vertices[1]->x(),
                                             triangle->vertices[2]->y() - triangle->vertices[1]->y(),
                                             triangle->vertices[2]->z() - triangle->vertices[1]->z()).norm());
    }
    if (numberOfHalfedge == 3)
    {
        return typename Boundary::Chain::Halfedge(index1, index3,
                                         gsVector3d<real_t>(
                                             triangle->vertices[0]->x() - triangle->vertices[2]->x(),
                                             triangle->vertices[0]->y() - triangle->vertices[2]->y(),
                                             triangle->vertices[0]->z() - triangle->vertices[2]->z()).norm());
    }
    return typename Boundary::Chain::Halfedge();
}

template<class T>
void gsHalfEdgeMesh<T>::sortVertices()
{
    std::size_t numberOfInnerVerticesFound = 0;
    std::vector<std::size_t> sorting(this->vertex.size(), 0);
    m_sorting = sorting;
    std::vector<std::size_t> inverseSorting(this->vertex.size(), 0);
    m_inverseSorting = inverseSorting;
    std::list<std::size_t> boundaryVertices = m_boundary.getVertexIndices();
    for (std::size_t i = 0; i < this->vertex.size(); i++)
    {
        if (!isBoundaryVertex(i))
        {
            numberOfInnerVerticesFound++;
            m_sorting[numberOfInnerVerticesFound - 1] = i;
            m_inverseSorting[i] = numberOfInnerVerticesFound;
        }
    }
    for (std::size_t i = 0; i < getNumberOfBoundaryVertices(); i++)
    {
        m_sorting[m_n + i] = boundaryVertices.front();
        m_inverseSorting[boundaryVertices.front()] = m_n + i + 1;
        boundaryVertices.pop_front();
    }
}

//***********************************************
//************ nested class Boundary ************
//***********************************************

template<class T>
gsHalfEdgeMesh<T>::Boundary::Boundary(const gismo::gsHalfEdgeMesh<T>::Boundary &boundary)
{
    m_boundary = boundary.m_boundary;
}

template<class T>
typename gsHalfEdgeMesh<T>::Boundary& gsHalfEdgeMesh<T>::Boundary::operator=(const gismo::gsHalfEdgeMesh<T>::Boundary &rhs)
{
    m_boundary = rhs.m_boundary;
    return *this;
}

template<class T>
gsHalfEdgeMesh<T>::Boundary::Boundary(const std::vector<typename gismo::gsHalfEdgeMesh<T>::Boundary::Chain::Halfedge> &halfedges)
{
    std::list<typename Chain::Halfedge> unsortedNonTwinHalfedges = findNonTwinHalfedges(halfedges);
    m_boundary.appendNextHalfedge(unsortedNonTwinHalfedges.front());
    unsortedNonTwinHalfedges.pop_front();
    std::queue<typename Chain::Halfedge> nonFittingHalfedges;
    while (!unsortedNonTwinHalfedges.empty())
    {
        if (m_boundary.isAppendableAsNext(unsortedNonTwinHalfedges.front()))
        {
            m_boundary.appendNextHalfedge(unsortedNonTwinHalfedges.front());
            unsortedNonTwinHalfedges.pop_front();
            while (!nonFittingHalfedges.empty())
            {
                unsortedNonTwinHalfedges.push_back(nonFittingHalfedges.front());
                nonFittingHalfedges.pop();
            }
        }
        else if (m_boundary.isAppendableAsPrev(unsortedNonTwinHalfedges.front()))
        {
            m_boundary.appendPrevHalfedge(unsortedNonTwinHalfedges.front());
            unsortedNonTwinHalfedges.pop_front();
            while (!nonFittingHalfedges.empty())
            {
                unsortedNonTwinHalfedges.push_back(nonFittingHalfedges.front());
                nonFittingHalfedges.pop();
            }
        }
        else
        {
            nonFittingHalfedges.push(unsortedNonTwinHalfedges.front());
            unsortedNonTwinHalfedges.pop_front();
        }
    }
    if (!m_boundary.isClosed())
        std::cout << "Warning: [" << __PRETTY_FUNCTION__
                  << "] Boundary is not closed although it should be. End points are: " << std::endl
                  << m_boundary.getFirstHalfedge().getOrigin() << std::endl << " and "
                  << m_boundary.getLastHalfedge().getEnd() << std::endl;
}

template<class T>
std::size_t gsHalfEdgeMesh<T>::Boundary::getNumberOfVertices() const
{
    return m_boundary.getNumberOfVertices();
}

template<class T>
real_t gsHalfEdgeMesh<T>::Boundary::getLength() const
{
    return m_boundary.getLength();
}

template<class T>
const std::vector<real_t> gsHalfEdgeMesh<T>::Boundary::getHalfedgeLengths() const
{
    return m_boundary.getHalfedgeLengths();
}

// TODO: refactor rest of inner classes

} // namespace gismo